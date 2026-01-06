"""
Tests for advanced covariance estimators.

Validates:
- Sample covariance baseline
- Ledoit-Wolf shrinkage (multiple targets)
- EWMA time-varying estimation
- RMT denoising via Marcenko-Pastur

Test Data:
- Random returns (controlled covariance structure)
- Edge cases (insufficient data, single asset, high-dimensional)
"""

import pytest
import numpy as np
import pandas as pd
from core.taa.covariance import (
    SampleEstimator,
    LedoitWolfEstimator,
    EWMAEstimator,
    RMTEstimator,
    estimate_covariance,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def random_returns():
    """Generate random returns with known covariance structure."""
    np.random.seed(42)
    n_periods = 120  # 10 years monthly
    n_assets = 5
    
    # Generate correlated returns
    true_corr = np.array([
        [1.0, 0.6, 0.3, 0.2, 0.1],
        [0.6, 1.0, 0.5, 0.3, 0.2],
        [0.3, 0.5, 1.0, 0.4, 0.3],
        [0.2, 0.3, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.3, 0.5, 1.0],
    ])
    
    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(true_corr)
    random_draws = np.random.randn(n_periods, n_assets)
    returns = random_draws @ L.T
    
    # Scale to realistic volatilities (5-20% annualized)
    vols = np.array([0.15, 0.10, 0.20, 0.12, 0.18]) / np.sqrt(12)  # Monthly
    returns = returns * vols
    
    df = pd.DataFrame(
        returns,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'],
        index=pd.date_range('2015-01-01', periods=n_periods, freq='M'),
    )
    
    return df


@pytest.fixture
def high_dimensional_returns():
    """Generate high-dimensional returns (N close to T)."""
    np.random.seed(42)
    n_periods = 60  # 5 years monthly
    n_assets = 40  # N/T = 0.67 (challenging regime)
    
    returns = np.random.randn(n_periods, n_assets) * 0.05  # 5% monthly vol
    df = pd.DataFrame(
        returns,
        columns=[f'Asset{i}' for i in range(n_assets)],
        index=pd.date_range('2020-01-01', periods=n_periods, freq='M'),
    )
    
    return df


@pytest.fixture
def vix_series():
    """Generate VIX time series with regime switches."""
    np.random.seed(42)
    n_periods = 120
    
    # Base VIX around 15-20
    vix = np.random.gamma(shape=2, scale=10, size=n_periods)
    
    # Add crisis periods (VIX > 30)
    crisis_periods = [20, 21, 22, 80, 81, 82, 83]
    vix[crisis_periods] = np.random.uniform(30, 50, size=len(crisis_periods))
    
    return pd.Series(
        vix,
        index=pd.date_range('2015-01-01', periods=n_periods, freq='M'),
    )


# =============================================================================
# SAMPLE ESTIMATOR TESTS
# =============================================================================

class TestSampleEstimator:
    """Test baseline sample covariance estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = SampleEstimator(min_periods=30)
        assert estimator.min_periods == 30
        assert estimator.covariance_ is None
    
    def test_invalid_min_periods(self):
        """Test error on invalid min_periods."""
        with pytest.raises(ValueError, match="min_periods must be >= 2"):
            SampleEstimator(min_periods=1)
    
    def test_fit_basic(self, random_returns):
        """Test basic covariance estimation."""
        estimator = SampleEstimator(min_periods=20)
        cov = estimator.fit(random_returns)
        
        # Check shape and symmetry
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= 0)
    
    def test_metadata(self, random_returns):
        """Test metadata reporting."""
        estimator = SampleEstimator(min_periods=20)
        estimator.fit(random_returns)
        metadata = estimator.get_metadata()
        
        assert metadata['method'] == 'sample'
        assert metadata['n_assets'] == 5
        assert metadata['n_periods'] == 120
        assert 'condition_number' in metadata
        assert 'min_eigenvalue' in metadata
        assert 'max_eigenvalue' in metadata
    
    def test_insufficient_data(self, random_returns):
        """Test error on insufficient data."""
        estimator = SampleEstimator(min_periods=150)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            estimator.fit(random_returns)


# =============================================================================
# LEDOIT-WOLF TESTS
# =============================================================================

class TestLedoitWolfEstimator:
    """Test Ledoit-Wolf shrinkage estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = LedoitWolfEstimator(
            shrinkage_target='constant_correlation',
            min_periods=30,
        )
        assert estimator.shrinkage_target == 'constant_correlation'
        assert estimator.min_periods == 30
    
    def test_invalid_target(self):
        """Test error on invalid shrinkage target."""
        with pytest.raises(ValueError, match="shrinkage_target must be one of"):
            LedoitWolfEstimator(shrinkage_target='invalid')
    
    def test_fit_identity_target(self, random_returns):
        """Test shrinkage toward identity matrix."""
        estimator = LedoitWolfEstimator(shrinkage_target='identity')
        cov = estimator.fit(random_returns)
        
        # Check shape and symmetry
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        
        # Check positive definite
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)
    
    def test_fit_constant_correlation(self, random_returns):
        """Test shrinkage toward constant correlation."""
        estimator = LedoitWolfEstimator(shrinkage_target='constant_correlation')
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        
        # Constant correlation should equalize off-diagonal correlations
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        off_diag = corr[~np.eye(5, dtype=bool)]
        
        # Standard deviation of correlations should be lower than sample
        assert off_diag.std() < 0.3  # Reasonable bound
    
    def test_fit_single_factor(self, random_returns):
        """Test shrinkage toward single-factor model."""
        estimator = LedoitWolfEstimator(shrinkage_target='single_factor')
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    
    def test_metadata(self, random_returns):
        """Test metadata reporting."""
        estimator = LedoitWolfEstimator(shrinkage_target='constant_correlation')
        estimator.fit(random_returns)
        metadata = estimator.get_metadata()
        
        assert metadata['method'] == 'ledoit_wolf'
        assert metadata['shrinkage_target'] == 'constant_correlation'
        assert 'shrinkage_intensity' in metadata
        assert 0.0 <= metadata['shrinkage_intensity'] <= 1.0
        assert metadata['n_assets'] == 5
        assert metadata['n_periods'] == 120
    
    def test_high_dimensional(self, high_dimensional_returns):
        """Test estimator on high-dimensional data (N close to T)."""
        estimator = LedoitWolfEstimator(shrinkage_target='constant_correlation')
        cov = estimator.fit(high_dimensional_returns)
        
        # Should produce valid covariance even with N/T = 0.67
        assert cov.shape == (40, 40)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)
        
        # High shrinkage expected when N/T is large
        metadata = estimator.get_metadata()
        assert metadata['shrinkage_intensity'] > 0.3  # Significant shrinkage


# =============================================================================
# EWMA TESTS
# =============================================================================

class TestEWMAEstimator:
    """Test EWMA covariance estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = EWMAEstimator(
            halflife=60,
            vix_threshold=25,
            fast_halflife=20,
        )
        assert estimator.halflife == 60
        assert estimator.vix_threshold == 25
        assert estimator.fast_halflife == 20
    
    def test_invalid_halflife(self):
        """Test error on invalid halflife."""
        with pytest.raises(ValueError, match="halflife must be > 0"):
            EWMAEstimator(halflife=-10)
        
        with pytest.raises(ValueError, match="fast_halflife must be in"):
            EWMAEstimator(halflife=60, fast_halflife=80)
    
    def test_fit_without_vix(self, random_returns):
        """Test EWMA estimation without VIX (no regime switching)."""
        estimator = EWMAEstimator(halflife=60, min_periods=20)
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    
    def test_fit_with_vix(self, random_returns, vix_series):
        """Test EWMA with VIX regime detection."""
        estimator = EWMAEstimator(halflife=60, vix_threshold=25, fast_halflife=20)
        cov = estimator.fit(random_returns, vix_series=vix_series)
        
        assert cov.shape == (5, 5)
        
        # Check metadata shows regime switches
        metadata = estimator.get_metadata()
        assert metadata['fast_periods_pct'] > 0  # Should detect crisis periods
        assert metadata['effective_halflife'] < metadata['halflife']  # Faster decay overall
    
    def test_metadata(self, random_returns):
        """Test metadata reporting."""
        estimator = EWMAEstimator(halflife=60)
        estimator.fit(random_returns)
        metadata = estimator.get_metadata()
        
        assert metadata['method'] == 'ewma'
        assert metadata['halflife'] == 60
        assert 'effective_halflife' in metadata
        assert metadata['n_assets'] == 5
        assert metadata['n_periods'] == 120


# =============================================================================
# RMT TESTS
# =============================================================================

class TestRMTEstimator:
    """Test Random Matrix Theory denoising estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = RMTEstimator(denoise_method='targeted_shrink')
        assert estimator.denoise_method == 'targeted_shrink'
    
    def test_invalid_method(self):
        """Test error on invalid denoise method."""
        with pytest.raises(ValueError, match="denoise_method must be one of"):
            RMTEstimator(denoise_method='invalid')
    
    def test_fit_targeted_shrink(self, random_returns):
        """Test RMT denoising with targeted shrinkage."""
        estimator = RMTEstimator(denoise_method='targeted_shrink')
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    
    def test_fit_constant(self, random_returns):
        """Test RMT denoising with constant method."""
        estimator = RMTEstimator(denoise_method='constant')
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
    
    def test_fit_clip(self, random_returns):
        """Test RMT denoising with clipping."""
        estimator = RMTEstimator(denoise_method='clip')
        cov = estimator.fit(random_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)
    
    def test_metadata(self, random_returns):
        """Test metadata reporting."""
        estimator = RMTEstimator(denoise_method='targeted_shrink')
        estimator.fit(random_returns)
        metadata = estimator.get_metadata()
        
        assert metadata['method'] == 'rmt'
        assert metadata['denoise_method'] == 'targeted_shrink'
        assert 'n_noise_eigenvalues' in metadata
        assert 'n_signal_eigenvalues' in metadata
        assert 'lambda_min' in metadata
        assert 'lambda_max' in metadata
        assert 'q_ratio' in metadata
        
        # N=5, T=120 → q=0.042, should have few noise eigenvalues
        assert metadata['q_ratio'] < 0.1
    
    def test_high_dimensional(self, high_dimensional_returns):
        """Test RMT on high-dimensional data (should denoise significantly)."""
        estimator = RMTEstimator(denoise_method='targeted_shrink')
        cov = estimator.fit(high_dimensional_returns)
        
        assert cov.shape == (40, 40)
        
        # With N=40, T=60 (q=0.67), expect many noise eigenvalues
        metadata = estimator.get_metadata()
        assert metadata['n_noise_eigenvalues'] > 10  # Significant noise
        assert metadata['q_ratio'] > 0.5
    
    def test_eigenvalue_denoising(self, high_dimensional_returns):
        """Test that eigenvalues are actually denoised."""
        estimator = RMTEstimator(denoise_method='targeted_shrink')
        estimator.fit(high_dimensional_returns)
        metadata = estimator.get_metadata()
        
        eigenvalues_original = np.array(metadata['eigenvalues_original'])
        eigenvalues_denoised = np.array(metadata['eigenvalues_denoised'])
        
        # Noise eigenvalues should be modified
        n_noise = metadata['n_noise_eigenvalues']
        if n_noise > 0:
            # Some eigenvalues should change
            assert not np.allclose(eigenvalues_original, eigenvalues_denoised)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Test estimate_covariance convenience function."""
    
    def test_sample_method(self, random_returns):
        """Test convenience function with sample method."""
        cov, metadata = estimate_covariance(random_returns, method='sample')
        
        assert cov.shape == (5, 5)
        assert metadata['method'] == 'sample'
    
    def test_ledoit_wolf_method(self, random_returns):
        """Test convenience function with Ledoit-Wolf."""
        cov, metadata = estimate_covariance(
            random_returns,
            method='ledoit_wolf',
            shrinkage_target='constant_correlation',
        )
        
        assert cov.shape == (5, 5)
        assert metadata['method'] == 'ledoit_wolf'
        assert metadata['shrinkage_target'] == 'constant_correlation'
    
    def test_ewma_method(self, random_returns, vix_series):
        """Test convenience function with EWMA."""
        cov, metadata = estimate_covariance(
            random_returns,
            method='ewma',
            vix_series=vix_series,
            halflife=60,
        )
        
        assert cov.shape == (5, 5)
        assert metadata['method'] == 'ewma'
    
    def test_rmt_method(self, random_returns):
        """Test convenience function with RMT."""
        cov, metadata = estimate_covariance(
            random_returns,
            method='rmt',
            denoise_method='targeted_shrink',
        )
        
        assert cov.shape == (5, 5)
        assert metadata['method'] == 'rmt'
    
    def test_invalid_method(self, random_returns):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_covariance(random_returns, method='invalid')


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestEstimatorComparison:
    """Compare estimators on same data."""
    
    def test_shrinkage_reduces_condition_number(self, random_returns):
        """Test that shrinkage improves condition number."""
        sample_est = SampleEstimator()
        sample_est.fit(random_returns)
        sample_cond = sample_est.get_metadata()['condition_number']
        
        lw_est = LedoitWolfEstimator(shrinkage_target='constant_correlation')
        lw_est.fit(random_returns)
        lw_cond = lw_est.get_metadata()['condition_number']
        
        # Ledoit-Wolf should have better condition number
        assert lw_cond <= sample_cond
    
    def test_rmt_reduces_noise_eigenvalues(self, high_dimensional_returns):
        """Test that RMT denoising reduces small eigenvalues."""
        sample_est = SampleEstimator()
        sample_cov = sample_est.fit(high_dimensional_returns)
        sample_eigenvalues = np.linalg.eigvalsh(sample_cov)
        
        rmt_est = RMTEstimator(denoise_method='targeted_shrink')
        rmt_cov = rmt_est.fit(high_dimensional_returns)
        rmt_eigenvalues = np.linalg.eigvalsh(rmt_cov)
        
        # RMT should have fewer very small eigenvalues
        sample_min = sample_eigenvalues.min()
        rmt_min = rmt_eigenvalues.min()
        
        assert rmt_min >= sample_min  # Denoising should raise minimum eigenvalue
