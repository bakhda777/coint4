"""Minimal Bybit v5 REST client for demo/paper trading."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode

import requests


class BybitRequestError(RuntimeError):
    """Raised when Bybit API responds with an error."""

    def __init__(self, message: str, ret_code: Optional[int] = None, ret_msg: Optional[str] = None):
        super().__init__(message)
        self.ret_code = ret_code
        self.ret_msg = ret_msg


@dataclass
class BybitSettings:
    """Connection settings for Bybit v5 API."""

    api_key: str
    api_secret: str
    base_url: str
    recv_window: int = 5000
    category: str = "linear"
    account_type: str = "UNIFIED"


def resolve_base_url(environment: str, override: Optional[str] = None) -> str:
    """Resolve base URL for Bybit environment."""
    if override:
        return override.rstrip("/")
    env = environment.lower().strip()
    if env == "demo":
        return "https://api-demo.bybit.com"
    if env == "testnet":
        return "https://api-testnet.bybit.com"
    return "https://api.bybit.com"


def map_timeframe_to_interval(timeframe: str) -> str:
    """Map config timeframe to Bybit kline interval."""
    normalized = timeframe.strip().lower()
    mapping = {
        "1min": "1",
        "1m": "1",
        "3min": "3",
        "3m": "3",
        "5min": "5",
        "5m": "5",
        "15min": "15",
        "15m": "15",
        "15t": "15",
        "30min": "30",
        "30m": "30",
        "1h": "60",
        "60min": "60",
        "60m": "60",
        "4h": "240",
        "240min": "240",
        "240m": "240",
        "1d": "D",
        "1day": "D",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def clamp_to_step(value: float, step: float) -> float:
    """Clamp value down to step size."""
    if step <= 0:
        return value
    steps = int(value / step)
    return steps * step


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class BybitRestClient:
    """Minimal Bybit REST client with v5 signing."""

    def __init__(
        self,
        settings: BybitSettings,
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.timeout = 15
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)

    def _sign(self, payload: str, timestamp: str) -> str:
        message = f"{timestamp}{self.settings.api_key}{self.settings.recv_window}{payload}"
        signature = hmac.new(
            self.settings.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        signed: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        method_upper = method.upper()
        retryable_http = {429, 500, 502, 503, 504}
        retryable_ret = {10006}

        for attempt in range(self.max_retries + 1):
            query = urlencode(params or {})
            payload = query if method_upper == "GET" else _json_dumps(body or {})

            headers = {"Content-Type": "application/json"}
            if signed:
                timestamp = str(int(time.time() * 1000))
                signature = self._sign(payload, timestamp)
                headers.update(
                    {
                        "X-BAPI-API-KEY": self.settings.api_key,
                        "X-BAPI-SIGN": signature,
                        "X-BAPI-SIGN-TYPE": "2",
                        "X-BAPI-TIMESTAMP": timestamp,
                        "X-BAPI-RECV-WINDOW": str(self.settings.recv_window),
                    }
                )

            try:
                if method_upper == "GET":
                    response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                else:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=None,
                        data=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise BybitRequestError(f"Request failed after retries: {exc}") from exc

            if response.status_code in retryable_http and attempt < self.max_retries:
                time.sleep(self.backoff_seconds * (2**attempt))
                continue
            if response.status_code >= 400:
                raise BybitRequestError(f"HTTP {response.status_code}: {response.text}")

            try:
                data = response.json()
            except ValueError as exc:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise BybitRequestError("Invalid JSON response") from exc

            ret_code = data.get("retCode", 0)
            if ret_code == 0:
                return data
            if ret_code in retryable_ret and attempt < self.max_retries:
                time.sleep(self.backoff_seconds * (2**attempt))
                continue
            ret_msg = data.get("retMsg")
            raise BybitRequestError(f"Bybit error {ret_code}: {ret_msg}", ret_code=ret_code, ret_msg=ret_msg)

        raise BybitRequestError("Retry loop exhausted")

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list[Dict[str, Any]]:
        params = {
            "category": self.settings.category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        data = self._request("GET", "/v5/market/kline", params=params, signed=False)
        raw = data.get("result", {}).get("list", [])
        rows = []
        for entry in raw:
            rows.append(
                {
                    "start_time_ms": int(entry[0]),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                    "turnover": float(entry[6]) if len(entry) > 6 else 0.0,
                }
            )
        rows.sort(key=lambda item: item["start_time_ms"])
        return rows

    def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        params = {"category": self.settings.category, "symbol": symbol}
        data = self._request("GET", "/v5/market/instruments-info", params=params, signed=False)
        items = data.get("result", {}).get("list", [])
        if not items:
            raise BybitRequestError(
                f"Missing instrument info for {symbol}",
                ret_code=10001,
                ret_msg="symbol invalid",
            )
        return items[0]

    def get_positions(self, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        params = {"category": self.settings.category}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/v5/position/list", params=params, signed=True)
        return data.get("result", {}).get("list", [])

    def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, Any]:
        params = {"accountType": self.settings.account_type, "coin": coin}
        data = self._request("GET", "/v5/account/wallet-balance", params=params, signed=True)
        items = data.get("result", {}).get("list", [])
        return items[0] if items else {}

    def create_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_idx: int,
        reduce_only: bool,
        order_link_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "category": self.settings.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "positionIdx": position_idx,
            "reduceOnly": reduce_only,
        }
        if order_link_id:
            payload["orderLinkId"] = order_link_id
        data = self._request("POST", "/v5/order/create", body=payload, signed=True)
        return data.get("result", {})

    def batch_get_instruments(
        self, symbols: Iterable[str], allow_missing: bool = False
    ) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
        info: Dict[str, Dict[str, Any]] = {}
        missing: list[str] = []
        for symbol in symbols:
            try:
                info[symbol] = self.get_instrument_info(symbol)
            except BybitRequestError as exc:
                if allow_missing and exc.ret_code == 10001:
                    missing.append(symbol)
                    continue
                raise
        return info, missing
