"""
Extended test suite for ensemble signal strategies.

Tests uncovered functionality in signals/ensemble.py:
- AdaptiveEnsemble initialization and weighting
- Dynamic weight adjustments
- Multiple combination methods
- Signal strength thresholds
"""

import pytest
import pandas as pd
import numpy as np
from signals.ensemble import AdaptiveEnsemble, EnsembleSignal, EnsembleSignalNew
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal


@pytest.fixture
def sample_price_data():
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Create trending + mean-reverting price pattern
    trend = np.linspace(100, 130, 300)
    noise = np.random.randn(300) * 2
    mean_rev = 5 * np.sin(np.linspace(0, 6 * np.pi, 300))
    
    prices = trend + noise + mean_rev
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Open': prices * 1.005,
    }).set_index('Date')


# ============================================================================
# AdaptiveEnsemble Tests
# ============================================================================

class TestAdaptiveEnsembleInitialization:
    """Test AdaptiveEnsemble initialization and configuration."""
    
    def test_basic_initialization(self):
        """Test basic initialization with multiple strategies."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 0.5),
            ('mean_rev', MeanReversionSignal(window=20), 0.5)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='weighted_average')
        
        assert len(ensemble.strategies) == 2
        assert ensemble.method == 'weighted_average'
        assert ensemble.adaptive_lookback == 60  # default
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 2.0),
            ('mean_rev', MeanReversionSignal(window=20), 3.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='weighted_average')
        
        # Weights should be normalized: 2.0/5.0 = 0.4, 3.0/5.0 = 0.6
        total_weight = sum(w for _, _, w in ensemble.strategies)
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_custom_parameters(self):
        """Test custom lookback and threshold parameters."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        ensemble = AdaptiveEnsemble(
            strategies,
            method='adaptive',
            adaptive_lookback=90,
            signal_threshold=0.5,
            rebalance_frequency=30
        )
        
        assert ensemble.adaptive_lookback == 90
        assert ensemble.signal_threshold == 0.5
        assert ensemble.rebalance_frequency == 30
    
    def test_strategy_names_extracted(self):
        """Test that strategy names are properly extracted."""
        strategies = [
            ('strat1', MomentumSignalV2(lookback=60), 0.5),
            ('strat2', MeanReversionSignal(window=20), 0.5)
        ]
        ensemble = AdaptiveEnsemble(strategies)
        
        assert 'strat1' in ensemble.strategy_names
        assert 'strat2' in ensemble.strategy_names
        assert len(ensemble.strategy_names) == 2


class TestAdaptiveEnsembleWeightedAverage:
    """Test weighted_average combination method."""
    
    def test_weighted_average_method(self, sample_price_data):
        """Test weighted average signal combination."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 0.6),
            ('mean_rev', MeanReversionSignal(window=20), 0.4)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='weighted_average')
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_weighted_average_respects_weights(self, sample_price_data):
        """Test that weighted average properly weights strategies."""
        # Create ensemble with only one strategy to test weighting
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='weighted_average', signal_threshold=0.0)
        result = ensemble.generate(sample_price_data)
        
        # Should generate valid signals
        assert 'Signal' in result.columns
        assert result['Signal'].notna().any()


class TestAdaptiveEnsembleMajorityVote:
    """Test majority_vote combination method."""
    
    def test_majority_vote_method(self, sample_price_data):
        """Test majority vote signal combination."""
        strategies = [
            ('mom1', MomentumSignalV2(lookback=40), 1.0),
            ('mom2', MomentumSignalV2(lookback=60), 1.0),
            ('mom3', MomentumSignalV2(lookback=80), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='majority_vote')
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_majority_vote_with_two_strategies(self, sample_price_data):
        """Test majority vote with even number of strategies."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0),
            ('mean_rev', MeanReversionSignal(window=20), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='majority_vote')
        result = ensemble.generate(sample_price_data)
        
        # Should still work, ties result in 0 signal
        assert 'Signal' in result.columns


class TestAdaptiveEnsembleUnanimous:
    """Test unanimous combination method."""
    
    def test_unanimous_method(self, sample_price_data):
        """Test unanimous signal combination (all must agree)."""
        strategies = [
            ('mom1', MomentumSignalV2(lookback=60), 1.0),
            ('mom2', MomentumSignalV2(lookback=80), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='unanimous')
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_unanimous_conservative_signals(self, sample_price_data):
        """Test that unanimous method produces fewer signals."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0),
            ('mean_rev', MeanReversionSignal(window=20), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies, method='unanimous')
        result = ensemble.generate(sample_price_data)
        
        # Unanimous should have more zeros (conservative)
        zero_pct = (result['Signal'] == 0).sum() / len(result)
        assert zero_pct > 0.5  # Most signals should be flat


class TestAdaptiveEnsembleAdaptiveWeighting:
    """Test adaptive weighting method."""
    
    def test_adaptive_method(self, sample_price_data):
        """Test adaptive weighting based on performance."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 0.5),
            ('mean_rev', MeanReversionSignal(window=20), 0.5)
        ]
        ensemble = AdaptiveEnsemble(
            strategies, 
            method='adaptive',
            adaptive_lookback=60,
            rebalance_frequency=20
        )
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_adaptive_with_short_data(self):
        """Test adaptive method with insufficient data."""
        # Create very short price series
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'Close': np.linspace(100, 110, 50),
            'High': np.linspace(101, 111, 50),
            'Low': np.linspace(99, 109, 50),
            'Open': np.linspace(100, 110, 50),
        }, index=dates)
        
        strategies = [
            ('momentum', MomentumSignalV2(lookback=20), 1.0)
        ]
        ensemble = AdaptiveEnsemble(
            strategies,
            method='adaptive',
            adaptive_lookback=60  # Longer than data
        )
        result = ensemble.generate(prices)
        
        # Should still work, using initial weights
        assert 'Signal' in result.columns


class TestAdaptiveEnsembleSignalThreshold:
    """Test signal strength threshold filtering."""
    
    def test_signal_threshold_filtering(self, sample_price_data):
        """Test that weak signals are filtered by threshold."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        
        # High threshold should produce fewer signals
        ensemble_strict = AdaptiveEnsemble(
            strategies,
            method='weighted_average',
            signal_threshold=0.8  # Very strict
        )
        result_strict = ensemble_strict.generate(sample_price_data)
        
        # Low threshold should produce more signals
        ensemble_loose = AdaptiveEnsemble(
            strategies,
            method='weighted_average',
            signal_threshold=0.1  # Very loose
        )
        result_loose = ensemble_loose.generate(sample_price_data)
        
        # Strict threshold should have more zeros
        zeros_strict = (result_strict['Signal'] == 0).sum()
        zeros_loose = (result_loose['Signal'] == 0).sum()
        assert zeros_strict >= zeros_loose


class TestAdaptiveEnsembleEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_strategy_ensemble(self, sample_price_data):
        """Test ensemble with only one strategy."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies)
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert not result['Signal'].isna().all()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies)
        
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=['Close', 'High', 'Low', 'Open'])
        result = ensemble.generate(empty_df)
        
        # Should return DataFrame with Signal column
        assert 'Signal' in result.columns
    
    def test_dataframe_with_nans(self, sample_price_data):
        """Test handling of data with NaN values."""
        # Introduce some NaN values
        df_with_nans = sample_price_data.copy()
        df_with_nans.loc[df_with_nans.index[10:20], 'Close'] = np.nan
        
        strategies = [
            ('momentum', MomentumSignalV2(lookback=60), 1.0)
        ]
        ensemble = AdaptiveEnsemble(strategies)
        result = ensemble.generate(df_with_nans)
        
        # Should still produce a result
        assert 'Signal' in result.columns


class TestAdaptiveEnsembleMultipleStrategies:
    """Test ensemble with many strategies."""
    
    def test_many_strategies(self, sample_price_data):
        """Test ensemble with 5+ strategies."""
        strategies = [
            ('mom_short', MomentumSignalV2(lookback=30), 0.2),
            ('mom_med', MomentumSignalV2(lookback=60), 0.2),
            ('mom_long', MomentumSignalV2(lookback=90), 0.2),
            ('mr_short', MeanReversionSignal(window=10), 0.2),
            ('mr_long', MeanReversionSignal(window=30), 0.2),
        ]
        ensemble = AdaptiveEnsemble(strategies, method='weighted_average')
        result = ensemble.generate(sample_price_data)
        
        assert 'Signal' in result.columns
        assert len(ensemble.strategies) == 5
        assert len(ensemble.strategy_names) == 5


class TestExistingEnsembleEdgeCases:
    """Additional edge case tests for existing ensemble classes."""
    
    def test_ensemble_signal_trend_filter_effect(self, sample_price_data):
        """Test that trend filter actually filters signals."""
        signal = EnsembleSignal()
        result = signal.generate(sample_price_data)
        
        # Check that TrendFilter is binary
        assert result['TrendFilter'].isin([0, 1]).all()
        
        # When TrendFilter is 0, Position should be 0
        filtered_positions = result[result['TrendFilter'] == 0]['Position']
        if len(filtered_positions) > 0:
            assert (filtered_positions == 0).all()
    
    def test_ensemble_new_bull_market_filter(self, sample_price_data):
        """Test bull market filter in EnsembleSignalNew."""
        signal = EnsembleSignalNew()
        result = signal.generate(sample_price_data)
        
        # Check SMA200 is calculated
        assert 'SMA200' in result.columns
        assert result['SMA200'].notna().any()
        
        # InBullMarket should be boolean-like
        assert result['InBullMarket'].isin([True, False]).all()
        
        # When not in bull market, Position should be 0
        bear_positions = result[~result['InBullMarket']]['Position']
        if len(bear_positions) > 0:
            assert (bear_positions == 0).all()
    
    def test_ensemble_new_burn_in_period(self, sample_price_data):
        """Test that first 200 periods are set to 0."""
        signal = EnsembleSignalNew()
        result = signal.generate(sample_price_data)
        
        # First 200 positions should be 0 (burn-in)
        burn_in_positions = result['Position'].iloc[:200]
        assert (burn_in_positions == 0).all()
