"""
Advanced Covariance Estimators for TAA Portfolio Optimization.

Implements production-ready covariance estimation methods to address
noise, non-stationarity, and estimation error in sample covariance matrices.

Methods Available:
- LedoitWolfEstimator: Shrinkage toward structured targets (identity, constant correlation)
- EWMAEstimator: Exponentially weighted moving average (time-varying, regime-aware)
- RMTEstimator: Random Matrix Theory denoising via Marcenko-Pastur distribution
- SampleEstimator: Baseline sample covariance (for comparison)

References:
- Ledoit & Wolf (2004): "Honey, I Shrunk the Sample Covariance Matrix"
- de Prado (2018): "Advances in Financial Machine Learning", Chapter 2 (RMT)
- RiskMetrics (1996): "Technical Document" (EWMA)

Usage:
    from core.taa.covariance import LedoitWolfEstimator, EWMAEstimator
    
    # Ledoit-Wolf shrinkage
    estimator = LedoitWolfEstimator(shrinkage_target='constant_correlation')
    cov_matrix = estimator.fit(returns_df)
    
    # EWMA (regime-aware)
    estimator = EWMAEstimator(halflife=60, vix_threshold=25)
    cov_matrix = estimator.fit(returns_df, vix_series=vix)
    
    # RMT denoising
    estimator = RMTEstimator(denoise_method='targeted_shrink')
    cov_matrix = estimator.fit(returns_df)
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple
import logging
from sklearn.covariance import LedoitWolf
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


class BaseCovarianceEstimator:
    """
    Base class for covariance estimators.
    
    All estimators implement:
    - fit(returns: pd.DataFrame) -> np.ndarray
    - get_metadata() -> dict
    """
    
    def __init__(self):
        self.covariance_ = None
        self.metadata_ = {}
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Fit covariance matrix to returns data.
        
        Args:
            returns: DataFrame with returns (rows=dates, cols=assets)
        
        Returns:
            Covariance matrix (N x N)
        
        Raises:
            ValueError: If returns has insufficient data
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def get_metadata(self) -> dict:
        """Get estimation metadata (shrinkage intensity, condition number, etc.)."""
        return self.metadata_.copy()


# =============================================================================
# SAMPLE COVARIANCE (BASELINE)
# =============================================================================

class SampleEstimator(BaseCovarianceEstimator):
    """
    Baseline sample covariance estimator.
    
    Uses standard formula: Σ = (1/(T-1)) * X'X
    
    Warning: Noisy and unstable when T/N is small (<10).
    Use for comparison only. Prefer shrinkage methods.
    
    Args:
        min_periods: Minimum observations required per asset (default: 20)
    
    Example:
        >>> estimator = SampleEstimator(min_periods=30)
        >>> cov = estimator.fit(returns_df)
        >>> metadata = estimator.get_metadata()
        >>> print(f"Condition number: {metadata['condition_number']:.1f}")
    """
    
    def __init__(self, min_periods: int = 20):
        super().__init__()
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        self.min_periods = min_periods
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate sample covariance matrix."""
        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} rows < {self.min_periods} min_periods"
            )
        
        # Drop columns with insufficient data
        valid_cols = returns.count() >= self.min_periods
        if not valid_cols.all():
            logger.warning(
                f"Dropping {(~valid_cols).sum()} assets with < {self.min_periods} observations"
            )
            returns = returns.loc[:, valid_cols]
        
        # Compute sample covariance
        self.covariance_ = returns.cov().values
        
        # Compute condition number (measure of numerical stability)
        eigenvalues = np.linalg.eigvalsh(self.covariance_)
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        self.metadata_ = {
            'method': 'sample',
            'n_assets': len(returns.columns),
            'n_periods': len(returns),
            'condition_number': condition_number,
            'min_eigenvalue': eigenvalues.min(),
            'max_eigenvalue': eigenvalues.max(),
        }
        
        return self.covariance_


# =============================================================================
# LEDOIT-WOLF SHRINKAGE
# =============================================================================

class LedoitWolfEstimator(BaseCovarianceEstimator):
    """
    Ledoit-Wolf shrinkage estimator with multiple target options.
    
    Shrinks sample covariance toward structured target:
        Σ_shrunk = δ * Σ_target + (1 - δ) * Σ_sample
    
    where δ (shrinkage intensity) is chosen to minimize expected error.
    
    Targets Available:
    - 'identity': Equal variance, zero correlation (diagonal matrix)
    - 'constant_correlation': Single correlation for all asset pairs
    - 'single_factor': Market model (one factor + idiosyncratic risk)
    
    Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix"
    
    Args:
        shrinkage_target: Target structure ('identity', 'constant_correlation', 'single_factor')
        min_periods: Minimum observations required (default: 20)
    
    Example:
        >>> estimator = LedoitWolfEstimator(shrinkage_target='constant_correlation')
        >>> cov = estimator.fit(returns_df)
        >>> metadata = estimator.get_metadata()
        >>> print(f"Shrinkage intensity: {metadata['shrinkage_intensity']:.2%}")
    """
    
    def __init__(
        self,
        shrinkage_target: Literal['identity', 'constant_correlation', 'single_factor'] = 'constant_correlation',
        min_periods: int = 20,
    ):
        super().__init__()
        valid_targets = ['identity', 'constant_correlation', 'single_factor']
        if shrinkage_target not in valid_targets:
            raise ValueError(
                f"shrinkage_target must be one of {valid_targets}, got '{shrinkage_target}'"
            )
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        
        self.shrinkage_target = shrinkage_target
        self.min_periods = min_periods
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate Ledoit-Wolf shrinkage covariance."""
        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} rows < {self.min_periods} min_periods"
            )
        
        # Drop columns with insufficient data
        valid_cols = returns.count() >= self.min_periods
        if not valid_cols.all():
            logger.warning(
                f"Dropping {(~valid_cols).sum()} assets with < {self.min_periods} observations"
            )
            returns = returns.loc[:, valid_cols]
        
        X = returns.values
        n_samples, n_features = X.shape
        
        # sklearn implementation (auto-selects optimal shrinkage)
        if self.shrinkage_target == 'identity':
            # Shrink toward identity (equal variance, zero correlation)
            lw = LedoitWolf(assume_centered=False)
            lw.fit(X)
            self.covariance_ = lw.covariance_
            shrinkage_intensity = lw.shrinkage_
        
        elif self.shrinkage_target == 'constant_correlation':
            # Shrink toward constant correlation
            self.covariance_, shrinkage_intensity = self._constant_correlation_shrinkage(X)
        
        else:  # single_factor
            # Shrink toward single-factor model
            self.covariance_, shrinkage_intensity = self._single_factor_shrinkage(X)
        
        # Compute diagnostics
        eigenvalues = np.linalg.eigvalsh(self.covariance_)
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        self.metadata_ = {
            'method': 'ledoit_wolf',
            'shrinkage_target': self.shrinkage_target,
            'shrinkage_intensity': shrinkage_intensity,
            'n_assets': n_features,
            'n_periods': n_samples,
            'condition_number': condition_number,
            'min_eigenvalue': eigenvalues.min(),
            'max_eigenvalue': eigenvalues.max(),
        }
        
        return self.covariance_
    
    def _constant_correlation_shrinkage(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Shrink toward constant correlation matrix."""
        n_samples, n_features = X.shape
        
        # Sample covariance
        S = np.cov(X, rowvar=False, bias=False)
        
        # Target: constant correlation
        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        corr = S / np.outer(sqrt_var, sqrt_var)
        avg_corr = (corr.sum() - n_features) / (n_features * (n_features - 1))
        
        # Constant correlation matrix
        F = np.ones((n_features, n_features)) * avg_corr
        np.fill_diagonal(F, 1.0)
        F = F * np.outer(sqrt_var, sqrt_var)
        
        # Optimal shrinkage intensity (Ledoit-Wolf formula)
        delta = self._compute_shrinkage_intensity(X, S, F)
        delta = np.clip(delta, 0.0, 1.0)
        
        # Shrunk covariance
        Sigma_shrunk = delta * F + (1 - delta) * S
        
        return Sigma_shrunk, delta
    
    def _single_factor_shrinkage(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Shrink toward single-factor (market) model."""
        n_samples, n_features = X.shape
        
        # Sample covariance
        S = np.cov(X, rowvar=False, bias=False)
        
        # Single-factor model: regress each asset on equal-weight market
        market_return = X.mean(axis=1)
        betas = np.array([np.cov(X[:, i], market_return)[0, 1] / market_return.var() 
                         for i in range(n_features)])
        market_var = market_return.var()
        
        # Target: beta * market_var * beta' + diagonal residual variance
        F = np.outer(betas, betas) * market_var
        residual_var = np.diag(S) - betas**2 * market_var
        np.fill_diagonal(F, np.diag(F) + residual_var)
        
        # Optimal shrinkage intensity
        delta = self._compute_shrinkage_intensity(X, S, F)
        delta = np.clip(delta, 0.0, 1.0)
        
        # Shrunk covariance
        Sigma_shrunk = delta * F + (1 - delta) * S
        
        return Sigma_shrunk, delta
    
    def _compute_shrinkage_intensity(
        self, X: np.ndarray, S: np.ndarray, F: np.ndarray
    ) -> float:
        """
        Compute optimal shrinkage intensity (Ledoit-Wolf formula).
        
        Minimizes expected squared error: E[||Σ_shrunk - Σ_true||^2]
        """
        n_samples, n_features = X.shape
        
        # Center data
        X_centered = X - X.mean(axis=0)
        
        # Compute pi-hat (variance of sample covariance)
        pi_mat = np.zeros((n_features, n_features))
        for i in range(n_samples):
            x = X_centered[i:i+1].T
            pi_mat += (x @ x.T - S) ** 2
        pi_hat = pi_mat.sum() / n_samples
        
        # Compute rho-hat (variance of target)
        rho_hat = ((F - S) ** 2).sum()
        
        # Compute gamma-hat (variance reduction from shrinkage)
        gamma_hat = np.linalg.norm(S - F, 'fro') ** 2
        
        # Optimal shrinkage
        if gamma_hat == 0:
            return 1.0  # Sample = target, full shrinkage
        
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / n_samples))
        
        return delta


# =============================================================================
# EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA)
# =============================================================================

class EWMAEstimator(BaseCovarianceEstimator):
    """
    Exponentially Weighted Moving Average (EWMA) covariance estimator.
    
    Time-varying covariance with exponential decay:
        Σ_t = λ * Σ_{t-1} + (1 - λ) * r_t * r_t'
    
    where λ is the decay factor (higher = slower adaptation).
    
    Features:
    - Regime-aware: Switch to faster decay in high volatility (VIX > threshold)
    - Captures time-varying correlations (crisis vs. normal periods)
    - More responsive than sample covariance
    
    Reference: RiskMetrics (1996) Technical Document
    
    Args:
        halflife: Halflife in periods (default: 60 for monthly data, ~5 years)
                  Decay factor λ = 0.5^(1/halflife)
        vix_threshold: VIX level to trigger fast decay (default: 25)
        fast_halflife: Halflife during high volatility (default: 20, ~1.5 years)
        min_periods: Minimum observations to initialize (default: 20)
    
    Example:
        >>> estimator = EWMAEstimator(halflife=60, vix_threshold=25)
        >>> cov = estimator.fit(returns_df, vix_series=vix)
        >>> metadata = estimator.get_metadata()
        >>> print(f"Effective halflife: {metadata['effective_halflife']:.1f}")
    """
    
    def __init__(
        self,
        halflife: float = 60,
        vix_threshold: float = 25.0,
        fast_halflife: float = 20,
        min_periods: int = 20,
    ):
        super().__init__()
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        if fast_halflife <= 0 or fast_halflife >= halflife:
            raise ValueError(
                f"fast_halflife must be in (0, {halflife}), got {fast_halflife}"
            )
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        
        self.halflife = halflife
        self.vix_threshold = vix_threshold
        self.fast_halflife = fast_halflife
        self.min_periods = min_periods
        
        # Decay factors: λ = 0.5^(1/halflife)
        self.lambda_normal = 0.5 ** (1.0 / halflife)
        self.lambda_fast = 0.5 ** (1.0 / fast_halflife)
    
    def fit(
        self,
        returns: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Estimate EWMA covariance matrix.
        
        Args:
            returns: DataFrame with returns (rows=dates, cols=assets)
            vix_series: Optional Series with VIX values (for regime detection)
        
        Returns:
            EWMA covariance matrix (N x N)
        """
        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} rows < {self.min_periods} min_periods"
            )
        
        # Drop columns with insufficient data
        valid_cols = returns.count() >= self.min_periods
        if not valid_cols.all():
            logger.warning(
                f"Dropping {(~valid_cols).sum()} assets with < {self.min_periods} observations"
            )
            returns = returns.loc[:, valid_cols]
        
        X = returns.values
        n_samples, n_features = X.shape
        
        # Initialize with sample covariance of first min_periods
        Sigma = np.cov(X[:self.min_periods].T, bias=False)
        
        # Track regime switches
        n_fast_periods = 0
        
        # Iterate through time, updating EWMA
        for t in range(self.min_periods, n_samples):
            r_t = X[t:t+1].T  # Column vector (n_features x 1)
            
            # Select decay factor based on VIX regime
            if vix_series is not None and vix_series.iloc[t] > self.vix_threshold:
                lambda_t = self.lambda_fast
                n_fast_periods += 1
            else:
                lambda_t = self.lambda_normal
            
            # EWMA update: Σ_t = λ * Σ_{t-1} + (1 - λ) * r_t * r_t'
            Sigma = lambda_t * Sigma + (1 - lambda_t) * (r_t @ r_t.T)
        
        self.covariance_ = Sigma
        
        # Compute diagnostics
        eigenvalues = np.linalg.eigvalsh(self.covariance_)
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        # Effective halflife (weighted average of normal and fast)
        fast_pct = n_fast_periods / (n_samples - self.min_periods)
        effective_halflife = (1 - fast_pct) * self.halflife + fast_pct * self.fast_halflife
        
        self.metadata_ = {
            'method': 'ewma',
            'halflife': self.halflife,
            'fast_halflife': self.fast_halflife,
            'vix_threshold': self.vix_threshold,
            'effective_halflife': effective_halflife,
            'fast_periods_pct': fast_pct,
            'n_assets': n_features,
            'n_periods': n_samples,
            'condition_number': condition_number,
            'min_eigenvalue': eigenvalues.min(),
            'max_eigenvalue': eigenvalues.max(),
        }
        
        return self.covariance_


# =============================================================================
# RANDOM MATRIX THEORY (RMT) DENOISING
# =============================================================================

class RMTEstimator(BaseCovarianceEstimator):
    """
    Random Matrix Theory (RMT) covariance denoising via Marcenko-Pastur distribution.
    
    Identifies and removes noise eigenvalues from sample covariance matrix.
    
    Process:
    1. Compute eigenvalues of sample correlation matrix
    2. Identify noise band using Marcenko-Pastur bounds [λ_min, λ_max]
    3. Denoise: Set noise eigenvalues to constant or shrink toward average
    4. Reconstruct covariance matrix from denoised eigenvalues
    
    Marcenko-Pastur bounds:
        λ_min = σ^2 * (1 - sqrt(N/T))^2
        λ_max = σ^2 * (1 + sqrt(N/T))^2
    
    where σ^2 = variance of random matrix eigenvalues, N = assets, T = observations.
    
    Reference: de Prado (2018), "Advances in Financial ML", Chapter 2
    
    Args:
        denoise_method: How to handle noise eigenvalues
                        - 'targeted_shrink': Shrink to average of noise band
                        - 'constant': Set to constant (average of noise eigenvalues)
                        - 'clip': Clip to λ_max (preserves more variance)
        min_periods: Minimum observations required (default: 20)
    
    Example:
        >>> estimator = RMTEstimator(denoise_method='targeted_shrink')
        >>> cov = estimator.fit(returns_df)
        >>> metadata = estimator.get_metadata()
        >>> print(f"Noise eigenvalues: {metadata['n_noise_eigenvalues']}")
    """
    
    def __init__(
        self,
        denoise_method: Literal['targeted_shrink', 'constant', 'clip'] = 'targeted_shrink',
        min_periods: int = 20,
    ):
        super().__init__()
        valid_methods = ['targeted_shrink', 'constant', 'clip']
        if denoise_method not in valid_methods:
            raise ValueError(
                f"denoise_method must be one of {valid_methods}, got '{denoise_method}'"
            )
        if min_periods < 2:
            raise ValueError(f"min_periods must be >= 2, got {min_periods}")
        
        self.denoise_method = denoise_method
        self.min_periods = min_periods
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate RMT-denoised covariance matrix."""
        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} rows < {self.min_periods} min_periods"
            )
        
        # Drop columns with insufficient data
        valid_cols = returns.count() >= self.min_periods
        if not valid_cols.all():
            logger.warning(
                f"Dropping {(~valid_cols).sum()} assets with < {self.min_periods} observations"
            )
            returns = returns.loc[:, valid_cols]
        
        X = returns.values
        n_samples, n_features = X.shape
        
        # Compute sample correlation matrix (easier to work with than covariance)
        corr = np.corrcoef(X, rowvar=False)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(corr)
        eigenvalues = eigenvalues[::-1]  # Descending order
        eigenvectors = eigenvectors[:, ::-1]
        
        # Marcenko-Pastur bounds for random matrix
        q = n_features / n_samples  # N/T ratio
        sigma_sq = self._estimate_variance(eigenvalues, q)
        lambda_min = sigma_sq * (1 - np.sqrt(q)) ** 2
        lambda_max = sigma_sq * (1 + np.sqrt(q)) ** 2
        
        # Identify noise eigenvalues
        noise_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
        n_noise = noise_mask.sum()
        
        # Denoise eigenvalues
        eigenvalues_denoised = eigenvalues.copy()
        if n_noise > 0:
            if self.denoise_method == 'targeted_shrink':
                # Shrink to average of noise band
                eigenvalues_denoised[noise_mask] = eigenvalues[noise_mask].mean()
            
            elif self.denoise_method == 'constant':
                # Set to constant (average)
                constant_value = eigenvalues[noise_mask].mean()
                eigenvalues_denoised[noise_mask] = constant_value
            
            else:  # clip
                # Clip to lambda_max
                eigenvalues_denoised = np.clip(eigenvalues_denoised, None, lambda_max)
        
        # Reconstruct correlation matrix
        corr_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
        
        # Ensure symmetry and PSD (numerical precision)
        corr_denoised = (corr_denoised + corr_denoised.T) / 2
        np.fill_diagonal(corr_denoised, 1.0)
        
        # Convert back to covariance using sample standard deviations
        std_devs = returns.std().values
        self.covariance_ = corr_denoised * np.outer(std_devs, std_devs)
        
        # Diagnostics
        eigenvalues_final = np.linalg.eigvalsh(self.covariance_)
        condition_number = eigenvalues_final.max() / eigenvalues_final.min()
        
        self.metadata_ = {
            'method': 'rmt',
            'denoise_method': self.denoise_method,
            'n_assets': n_features,
            'n_periods': n_samples,
            'q_ratio': q,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'n_noise_eigenvalues': n_noise,
            'n_signal_eigenvalues': n_features - n_noise,
            'condition_number': condition_number,
            'min_eigenvalue': eigenvalues_final.min(),
            'max_eigenvalue': eigenvalues_final.max(),
            'eigenvalues_original': eigenvalues.tolist(),
            'eigenvalues_denoised': eigenvalues_denoised.tolist(),
        }
        
        return self.covariance_
    
    def _estimate_variance(self, eigenvalues: np.ndarray, q: float) -> float:
        """
        Estimate variance of random matrix eigenvalues.
        
        Use median eigenvalue as robust estimate of σ^2.
        """
        # de Prado method: use median as robust estimate
        sigma_sq = np.median(eigenvalues)
        return sigma_sq


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def estimate_covariance(
    returns: pd.DataFrame,
    method: Literal['sample', 'ledoit_wolf', 'ewma', 'rmt'] = 'ledoit_wolf',
    vix_series: Optional[pd.Series] = None,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function for covariance estimation.
    
    Args:
        returns: DataFrame with asset returns
        method: Estimation method ('sample', 'ledoit_wolf', 'ewma', 'rmt')
        vix_series: Optional VIX series for EWMA regime detection
        **kwargs: Method-specific parameters
    
    Returns:
        (covariance_matrix, metadata)
    
    Example:
        >>> cov, meta = estimate_covariance(returns, method='ledoit_wolf',
        ...                                  shrinkage_target='constant_correlation')
        >>> print(f"Shrinkage: {meta['shrinkage_intensity']:.2%}")
    """
    if method == 'sample':
        estimator = SampleEstimator(**kwargs)
    elif method == 'ledoit_wolf':
        estimator = LedoitWolfEstimator(**kwargs)
    elif method == 'ewma':
        estimator = EWMAEstimator(**kwargs)
    elif method == 'rmt':
        estimator = RMTEstimator(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if method == 'ewma' and vix_series is not None:
        cov_matrix = estimator.fit(returns, vix_series=vix_series)
    else:
        cov_matrix = estimator.fit(returns)
    
    return cov_matrix, estimator.get_metadata()
