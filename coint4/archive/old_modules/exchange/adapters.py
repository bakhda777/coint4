"""Exchange adapters for live trading."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseExchangeAdapter(ABC):
    """Base class for exchange adapters."""
    
    @abstractmethod
    def fetch_ohlcv(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for a pair."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    def place_order(self, pair: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place an order."""
        pass


class PaperTradingAdapter(BaseExchangeAdapter):
    """Paper trading adapter using local parquet files."""
    
    def __init__(self, data_path: str = "data_downloaded", initial_balance: float = 100000):
        """Initialize paper trading adapter.
        
        Args:
            data_path: Path to parquet data files
            initial_balance: Initial balance for paper trading
        """
        self.data_path = Path(data_path)
        self.balance = {"USDT": initial_balance}
        self.positions = {}
        self.orders = []
        self.last_prices = {}
        
    def fetch_ohlcv(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from local parquet files.
        
        Simulates live data by reading the most recent bars from parquet.
        """
        # Parse pair (e.g., "BTC/USDT" -> "BTCUSDT")
        symbol = pair.replace("/", "")
        
        # For demo, create synthetic data
        # In production, would read from actual parquet files
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15 * limit)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq='15T')
        
        # Generate synthetic OHLCV data
        np.random.seed(42)  # For reproducibility
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        prices = base_price * (1 + np.random.randn(len(timestamps)) * 0.001)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.randn(len(timestamps)) * 0.0005),
            'high': prices * (1 + np.abs(np.random.randn(len(timestamps))) * 0.001),
            'low': prices * (1 - np.abs(np.random.randn(len(timestamps))) * 0.001),
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(timestamps))
        })
        
        # Store last price
        self.last_prices[symbol] = df['close'].iloc[-1]
        
        logger.info(f"Fetched {len(df)} bars for {pair}")
        return df
    
    def get_balance(self) -> Dict[str, float]:
        """Get current balance including positions."""
        total_balance = self.balance.copy()
        
        # Add position values
        for symbol, amount in self.positions.items():
            if symbol in self.last_prices:
                value_usdt = amount * self.last_prices[symbol]
                total_balance["USDT"] += value_usdt
        
        return total_balance
    
    def place_order(self, pair: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Simulate order placement.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            amount: Amount to trade
            price: Limit price (None for market order)
        """
        symbol = pair.replace("/", "")
        
        # Get current price
        current_price = self.last_prices.get(symbol, 50000)
        execution_price = price if price else current_price
        
        order = {
            "id": len(self.orders) + 1,
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "side": side,
            "amount": amount,
            "price": execution_price,
            "status": "filled",  # Instant fill for paper trading
            "type": "limit" if price else "market"
        }
        
        # Update balance and positions
        cost = amount * execution_price
        
        if side == "buy":
            if self.balance["USDT"] >= cost:
                self.balance["USDT"] -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + amount
                logger.info(f"Bought {amount} {symbol} at {execution_price}")
            else:
                order["status"] = "rejected"
                order["reason"] = "Insufficient balance"
                logger.warning(f"Order rejected: insufficient balance")
        else:  # sell
            if self.positions.get(symbol, 0) >= amount:
                self.balance["USDT"] += cost
                self.positions[symbol] -= amount
                logger.info(f"Sold {amount} {symbol} at {execution_price}")
            else:
                order["status"] = "rejected"
                order["reason"] = "Insufficient position"
                logger.warning(f"Order rejected: insufficient position")
        
        self.orders.append(order)
        return order


class LiveExchangeAdapter(BaseExchangeAdapter):
    """Live exchange adapter (placeholder for real exchange integration)."""
    
    def __init__(self):
        """Initialize live exchange adapter.
        
        Would read API keys from environment variables.
        """
        self.api_key = os.getenv("EXCHANGE_API_KEY", "")
        self.api_secret = os.getenv("EXCHANGE_API_SECRET", "")
        
        if not self.api_key or not self.api_secret:
            logger.warning("Exchange API credentials not found in environment")
    
    def fetch_ohlcv(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch live OHLCV data from exchange."""
        raise NotImplementedError("Live exchange integration not implemented")
    
    def get_balance(self) -> Dict[str, float]:
        """Get live account balance."""
        raise NotImplementedError("Live exchange integration not implemented")
    
    def place_order(self, pair: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place live order on exchange."""
        raise NotImplementedError("Live exchange integration not implemented")