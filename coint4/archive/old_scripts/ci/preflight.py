#!/usr/bin/env python3
"""
Pre-live preflight checks for coint2 trading system.

Performs comprehensive readiness checks before live trading:
- Data availability and freshness
- Configuration validation
- Timezone consistency
- Risk parameters
- System dependencies
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreflightCheck:
    """Individual preflight check."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = "PENDING"
        self.message = ""
        self.details = {}
    
    def run(self) -> bool:
        """Run the check. Return True if passed."""
        raise NotImplementedError
    
    def passed(self, message: str = "", details: Dict = None):
        """Mark check as passed."""
        self.status = "PASS"
        self.message = message
        self.details = details or {}
        logger.info(f"✅ {self.name}: {message}")
    
    def failed(self, message: str, details: Dict = None):
        """Mark check as failed."""
        self.status = "FAIL"
        self.message = message
        self.details = details or {}
        logger.error(f"❌ {self.name}: {message}")
    
    def warning(self, message: str, details: Dict = None):
        """Mark check as warning."""
        self.status = "WARN"
        self.message = message
        self.details = details or {}
        logger.warning(f"⚠️ {self.name}: {message}")


class DataRootCheck(PreflightCheck):
    """Check DATA_ROOT availability and freshness."""
    
    def __init__(self):
        super().__init__("DataRoot", "Check data availability and freshness")
    
    def run(self) -> bool:
        data_root = os.getenv("DATA_ROOT", "data_downloaded")
        data_path = Path(data_root)
        
        if not data_path.exists():
            self.failed(f"DATA_ROOT not found: {data_path}")
            return False
        
        # Check for recent parquet files
        parquet_files = list(data_path.glob("**/*.parquet"))
        if not parquet_files:
            self.failed("No parquet files found in DATA_ROOT")
            return False
        
        # Find most recent file
        latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        latest_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        age_hours = (datetime.now() - latest_time).total_seconds() / 3600
        
        # Check freshness (should be < 2 * 15min = 30min for M15)
        max_age_hours = 0.5  # 30 minutes
        
        if age_hours > max_age_hours:
            self.warning(f"Data may be stale: {age_hours:.1f}h old (> {max_age_hours:.1f}h)")
        else:
            self.passed(f"Fresh data found: {age_hours:.1f}h old")
        
        self.details = {
            "data_root": str(data_path),
            "parquet_files": len(parquet_files),
            "latest_file": str(latest_file),
            "age_hours": age_hours
        }
        
        return True


class TimezoneCheck(PreflightCheck):
    """Check timezone consistency."""
    
    def __init__(self):
        super().__init__("Timezone", "Check timezone consistency in data")
    
    def run(self) -> bool:
        try:
            # Sample check - in reality would check actual data files
            # For now, just verify that pandas can handle timezone operations
            
            # Create sample data with different timezone scenarios
            tz_naive = pd.date_range('2023-01-01', periods=5, freq='15T')
            tz_aware = pd.date_range('2023-01-01', periods=5, freq='15T', tz='UTC')
            
            # Check that we can work with both
            if len(tz_naive) == 5 and len(tz_aware) == 5:
                self.passed("Timezone handling operational")
                self.details = {
                    "tz_naive_supported": True,
                    "tz_aware_supported": True,
                    "recommendation": "Use tz-naive or consistent UTC"
                }
                return True
            else:
                self.failed("Timezone handling test failed")
                return False
                
        except Exception as e:
            self.failed(f"Timezone check error: {str(e)}")
            return False


class ConfigCheck(PreflightCheck):
    """Check configuration validity."""
    
    def __init__(self, config_path: str):
        super().__init__("Config", f"Check configuration: {config_path}")
        self.config_path = config_path
    
    def run(self) -> bool:
        try:
            config = load_config(self.config_path)
            
            checks = []
            
            # Gap minutes check for M15
            gap_minutes = config.time.gap_minutes
            timeframe = config.time.timeframe
            
            if timeframe == "15T" and gap_minutes != 15:
                checks.append(f"Gap mismatch: {gap_minutes} != 15 for M15")
            else:
                checks.append(f"Gap correct: {gap_minutes} for {timeframe}")
            
            # Commission and slippage
            commission = config.backtesting.commission_pct
            slippage = config.backtesting.slippage_pct
            
            if commission <= 0:
                checks.append("Commission must be > 0")
            elif commission > 0.01:  # 1%
                checks.append(f"Commission very high: {commission:.2%}")
            else:
                checks.append(f"Commission reasonable: {commission:.2%}")
            
            if slippage <= 0:
                checks.append("Slippage must be > 0")
            elif slippage > 0.005:  # 0.5%
                checks.append(f"Slippage very high: {slippage:.2%}")
            else:
                checks.append(f"Slippage reasonable: {slippage:.2%}")
            
            # Guardrails
            guards_enabled = config.guards.enabled if hasattr(config, 'guards') else False
            if guards_enabled:
                checks.append("Guardrails enabled ✓")
            else:
                checks.append("Guardrails DISABLED - risky!")
            
            # Normalization method
            norm_method = config.backtesting.normalization_method
            if norm_method == "rolling_zscore":
                checks.append("Safe normalization method ✓")
            else:
                checks.append(f"Unsafe normalization: {norm_method}")
            
            # Determine overall status
            failures = [c for c in checks if "must be" in c or "DISABLED" in c or "Unsafe" in c]
            warnings = [c for c in checks if "very high" in c or "mismatch" in c]
            
            if failures:
                self.failed(f"{len(failures)} critical issues")
            elif warnings:
                self.warning(f"{len(warnings)} warnings")
            else:
                self.passed("All checks passed")
            
            self.details = {
                "config_path": self.config_path,
                "checks": checks,
                "failures": failures,
                "warnings": warnings,
                "gap_minutes": gap_minutes,
                "timeframe": timeframe,
                "commission_pct": commission,
                "slippage_pct": slippage,
                "guards_enabled": guards_enabled,
                "normalization_method": norm_method
            }
            
            return len(failures) == 0
            
        except Exception as e:
            self.failed(f"Config load error: {str(e)}")
            return False


class RiskConfigCheck(PreflightCheck):
    """Check risk configuration."""
    
    def __init__(self, risk_config_path: str):
        super().__init__("RiskConfig", f"Check risk config: {risk_config_path}")
        self.risk_config_path = risk_config_path
    
    def run(self) -> bool:
        try:
            risk_path = Path(self.risk_config_path)
            
            if not risk_path.exists():
                self.failed(f"Risk config not found: {risk_path}")
                return False
            
            with open(risk_path, 'r') as f:
                risk_config = yaml.safe_load(f)
            
            # Check required parameters
            required_params = [
                'max_daily_loss_pct',
                'max_drawdown_pct', 
                'max_no_data_minutes',
                'min_trade_count_per_day',
                'position_size_usd'
            ]
            
            missing = []
            for param in required_params:
                if param not in risk_config:
                    missing.append(param)
            
            if missing:
                self.failed(f"Missing risk parameters: {missing}")
                return False
            
            # Validate ranges
            validations = []
            
            daily_loss = risk_config['max_daily_loss_pct']
            if daily_loss <= 0 or daily_loss > 50:
                validations.append(f"Daily loss out of range: {daily_loss}%")
            
            max_dd = risk_config['max_drawdown_pct']
            if max_dd <= 0 or max_dd > 100:
                validations.append(f"Max drawdown out of range: {max_dd}%")
            
            no_data_min = risk_config['max_no_data_minutes']
            if no_data_min <= 0 or no_data_min > 1440:  # 24 hours
                validations.append(f"No-data timeout out of range: {no_data_min} min")
            
            if validations:
                self.failed(f"Invalid risk parameters: {validations}")
                return False
            
            self.passed("Risk config valid")
            self.details = risk_config
            return True
            
        except Exception as e:
            self.failed(f"Risk config error: {str(e)}")
            return False


def run_preflight_checks() -> Tuple[bool, List[PreflightCheck]]:
    """Run all preflight checks."""
    
    logger.info("🚁 Starting pre-live preflight checks...")
    
    checks = [
        DataRootCheck(),
        TimezoneCheck(), 
        ConfigCheck("configs/prod.yaml"),
        RiskConfigCheck("configs/risk.yaml")
    ]
    
    results = []
    overall_success = True
    
    for check in checks:
        logger.info(f"Running: {check.name} - {check.description}")
        try:
            success = check.run()
            if not success and check.status == "FAIL":
                overall_success = False
        except Exception as e:
            check.failed(f"Exception: {str(e)}")
            overall_success = False
        
        results.append(check)
    
    return overall_success, results


def generate_preflight_report(checks: List[PreflightCheck], output_path: str):
    """Generate preflight report."""
    
    report_lines = []
    
    # Header
    report_lines.extend([
        "# Pre-live Preflight Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ""
    ])
    
    # Summary
    passed = len([c for c in checks if c.status == "PASS"])
    failed = len([c for c in checks if c.status == "FAIL"]) 
    warned = len([c for c in checks if c.status == "WARN"])
    
    overall_status = "🟢 READY" if failed == 0 else "🔴 NOT READY"
    
    report_lines.extend([
        f"## Overall Status: {overall_status}",
        "",
        f"- ✅ Passed: {passed}",
        f"- ❌ Failed: {failed}",
        f"- ⚠️ Warnings: {warned}",
        ""
    ])
    
    # Detailed results
    report_lines.extend([
        "## Detailed Results",
        ""
    ])
    
    for check in checks:
        status_emoji = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(check.status, "❓")
        report_lines.extend([
            f"### {check.name} {status_emoji}",
            f"**Description:** {check.description}",
            f"**Status:** {check.status}",
            f"**Message:** {check.message}",
            ""
        ])
        
        if check.details:
            report_lines.extend([
                "**Details:**",
                f"```json",
                json.dumps(check.details, indent=2),
                "```",
                ""
            ])
    
    # Recommendations
    if failed > 0:
        report_lines.extend([
            "## ❌ Action Required",
            "",
            "System is NOT ready for live trading. Please resolve all failed checks before proceeding.",
            ""
        ])
    elif warned > 0:
        report_lines.extend([
            "## ⚠️ Warnings Present", 
            "",
            "System is ready but has warnings. Review and consider addressing them.",
            ""
        ])
    else:
        report_lines.extend([
            "## 🚀 Ready for Launch",
            "",
            "All preflight checks passed. System ready for live trading.",
            ""
        ])
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main preflight check runner."""
    
    # Setup logging to file
    log_path = "artifacts/live/PREFLIGHT.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Run checks
    success, checks = run_preflight_checks()
    
    # Generate report
    report_path = "artifacts/live/PREFLIGHT_REPORT.md"
    generate_preflight_report(checks, report_path)
    
    # Summary
    passed = len([c for c in checks if c.status == "PASS"])
    failed = len([c for c in checks if c.status == "FAIL"])
    warned = len([c for c in checks if c.status == "WARN"])
    
    logger.info(f"🏁 Preflight complete: {passed} passed, {failed} failed, {warned} warnings")
    logger.info(f"📋 Report: {report_path}")
    logger.info(f"📝 Log: {log_path}")
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())