#\!/usr/bin/env python3
"""DEPRECATED: Use optuna_report.py analyze"""
import warnings
warnings.warn("Use: python scripts/analysis/optuna_report.py analyze", DeprecationWarning)

import sys
from optuna_report import main
sys.exit(main(['analyze'] + sys.argv[1:]))
