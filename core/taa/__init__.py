"""
Tactical Asset Allocation (TAA) Module.

Provides production-ready optimizers and covariance estimators for TAA.

Optimizers:
- Mean-variance, Max Sharpe, Risk Parity, Black-Litterman, CVaR, HRP, Kelly

Covariance Estimators:
- Sample (baseline)
- Ledoit-Wolf shrinkage (multiple targets)
- EWMA (time-varying, regime-aware)
- RMT denoising (Marcenko-Pastur)

Usage:
    from core.taa import TAAOptimizer, estimate_covariance
    
    cov, meta = estimate_covariance(returns, method='ledoit_wolf')
    optimizer = TAAOptimizer(constraints, method='max_sharpe')
    weights, _ = optimizer.optimize(expected_returns, cov)
"""

from .optimizer import TAAOptimizer
from .constraints import OptimizationConstraints, load_constraints_from_config
from .covariance import (
    SampleEstimator,
    LedoitWolfEstimator,
    EWMAEstimator,
    RMTEstimator,
    estimate_covariance,
)

__all__ = [
    'TAAOptimizer',
    'OptimizationConstraints',
    'load_constraints_from_config',
    'SampleEstimator',
    'LedoitWolfEstimator',
    'EWMAEstimator',
    'RMTEstimator',
    'estimate_covariance',
]
