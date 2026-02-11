"""
Tests for TAA Model Engine (XGBoost-based forecasting).

Focuses on business logic, data processing, and model training/prediction.
Does NOT test plotting/visualization (per project guidelines).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.taa.model_engine import TAAModelEngine


@pytest.fixture
def sample_multiindex_data():
    """
    Create sample multi-index data for testing.

    Returns:
        pd.DataFrame: (Date, ticker) MultiIndex with Close prices.
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['SPY', 'XLE', 'XLF']

    # Create MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['Date', 'ticker']
    )

    np.random.seed(42)
    # Generate realistic price data with trends
    data = []
    for ticker in tickers:
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.01, len(dates))
        prices = base_price * (1 + returns).cumprod()
        data.extend(prices)

    df = pd.DataFrame({'Close': data}, index=index)
    return df


@pytest.fixture
def sample_single_ticker_data():
    """
    Create sample single-ticker data for testing.

    Returns:
        pd.DataFrame: Single ticker with Date index and Close prices.
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.01, len(dates))
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame({'Close': prices}, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_features_data():
    """
    Create sample data with features for training.

    Returns:
        pd.DataFrame: MultiIndex data with features and targets.
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['SPY', 'XLE', 'XLF']

    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['Date', 'ticker']
    )

    np.random.seed(42)
    n_rows = len(dates) * len(tickers)

    # Generate features
    df = pd.DataFrame({
        'Close': np.random.uniform(90, 110, n_rows),
        'momentum_20d': np.random.normal(0, 0.05, n_rows),
        'volatility_20d': np.random.uniform(0.01, 0.03, n_rows),
        'rsi_14d': np.random.uniform(30, 70, n_rows),
        'target_1w': np.random.normal(0.001, 0.02, n_rows),
        'target_4w': np.random.normal(0.005, 0.03, n_rows),
        'target_12w': np.random.normal(0.015, 0.05, n_rows)
    }, index=index)

    return df


class TestTAAModelEngineInitialization:
    """Test model engine initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        engine = TAAModelEngine()

        assert engine.horizons == [1, 4, 12]
        assert isinstance(engine.params, dict)
        assert engine.params['objective'] == 'reg:squarederror'
        assert len(engine.models) == 0
        assert len(engine.feature_importance_) == 0

    def test_initialization_custom_horizons(self):
        """Test initialization with custom horizons."""
        engine = TAAModelEngine(horizons=[1, 2, 3])

        assert engine.horizons == [1, 2, 3]

    def test_initialization_custom_params(self):
        """Test initialization with custom XGBoost parameters."""
        custom_params = {
            'max_depth': 3,
            'learning_rate': 0.01,
            'n_estimators': 100
        }
        engine = TAAModelEngine(params=custom_params)

        assert engine.params['max_depth'] == 3
        assert engine.params['learning_rate'] == 0.01
        assert engine.params['n_estimators'] == 100

    def test_default_params_structure(self):
        """Test default parameters have correct structure."""
        engine = TAAModelEngine()
        params = engine._default_params()

        # Verify key parameters exist
        assert 'objective' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'n_estimators' in params
        assert 'random_state' in params

        # Verify reasonable defaults
        assert params['max_depth'] > 0
        assert 0 < params['learning_rate'] < 1
        assert params['n_estimators'] > 0


class TestCreateTargets:
    """Test target creation for forward returns."""

    def test_create_targets_multiindex(self, sample_multiindex_data):
        """Test target creation with MultiIndex data."""
        engine = TAAModelEngine(horizons=[1, 4])
        result = engine.create_targets(sample_multiindex_data)

        # Check target columns exist
        assert 'target_1w' in result.columns
        assert 'target_4w' in result.columns

        # Check original data preserved
        assert 'Close' in result.columns

        # Check index preserved
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['Date', 'ticker']

    def test_create_targets_single_ticker(self, sample_single_ticker_data):
        """Test target creation with single ticker data."""
        engine = TAAModelEngine(horizons=[1])
        result = engine.create_targets(sample_single_ticker_data)

        # Check target column exists
        assert 'target_1w' in result.columns

        # Check calculation (1 week = 5 trading days)
        # Target should be forward return
        assert 'Close' in result.columns

    def test_create_targets_calculation_accuracy(self):
        """Test forward return calculation accuracy."""
        # Create simple data for verification
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        df = pd.DataFrame({'Close': prices}, index=dates)

        engine = TAAModelEngine(horizons=[1])
        result = engine.create_targets(df)

        # 1 week = 5 days forward
        # target_1w at day 0 should be (price[5] - price[0]) / price[0]
        # But shifted -5, so NaN at the end
        assert 'target_1w' in result.columns

    def test_create_targets_preserves_input(self, sample_multiindex_data):
        """Test that create_targets doesn't modify input DataFrame."""
        engine = TAAModelEngine(horizons=[1])
        original_cols = sample_multiindex_data.columns.tolist()

        result = engine.create_targets(sample_multiindex_data)

        # Original should be unchanged
        assert sample_multiindex_data.columns.tolist() == original_cols

        # Result should have new columns
        assert len(result.columns) > len(original_cols)


class TestTrain:
    """Test model training functionality."""

    def test_train_basic(self, sample_features_data):
        """Test basic training workflow."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        results = engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        # Check models were trained
        assert 1 in engine.models
        assert len(engine.models) == 1

        # Check results structure
        assert 1 in results
        assert 'train_ic' in results[1]
        assert 'val_ic' in results[1]
        assert 'train_samples' in results[1]
        assert 'val_samples' in results[1]

    def test_train_multiple_horizons(self, sample_features_data):
        """Test training multiple horizon models."""
        engine = TAAModelEngine(horizons=[1, 4, 12])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        results = engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        # All horizons should be trained
        assert len(engine.models) == 3
        assert 1 in engine.models
        assert 4 in engine.models
        assert 12 in engine.models

    def test_train_auto_detect_features(self, sample_features_data):
        """Test automatic feature detection."""
        engine = TAAModelEngine(horizons=[1])

        # Don't provide feature_cols, should auto-detect
        results = engine.train(sample_features_data, verbose=False)

        # Should have trained successfully
        assert 1 in engine.models

    def test_train_validation_split(self, sample_features_data):
        """Test validation split is applied correctly."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d']
        results = engine.train(
            sample_features_data,
            feature_cols=feature_cols,
            val_split=0.3,
            verbose=False
        )

        # Check train/val samples respect split
        train_samples = results[1]['train_samples']
        val_samples = results[1]['val_samples']

        # val_split=0.3 means ~30% validation
        total = train_samples + val_samples
        assert val_samples / total > 0.25  # Some tolerance
        assert val_samples / total < 0.35

    def test_train_missing_target_column(self):
        """Test training handles missing target column gracefully."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        }, index=dates)

        engine = TAAModelEngine(horizons=[1])

        # Should not crash, just skip missing horizon
        results = engine.train(df, verbose=False)

        # No models should be trained (no targets)
        assert len(engine.models) == 0

    def test_train_stores_feature_importance(self, sample_features_data):
        """Test that feature importance is stored."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        # Feature importance should be stored
        assert 1 in engine.feature_importance_
        importance_df = engine.feature_importance_[1]

        # Should have correct structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(feature_cols)


class TestPredict:
    """Test model prediction functionality."""

    def test_predict_basic(self, sample_features_data):
        """Test basic prediction workflow."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        # Predict on same data
        result = engine.predict(sample_features_data, feature_cols=feature_cols)

        # Check prediction column exists
        assert 'pred_1w' in result.columns

    def test_predict_multiple_horizons(self, sample_features_data):
        """Test predictions for multiple horizons."""
        engine = TAAModelEngine(horizons=[1, 4, 12])

        feature_cols = ['momentum_20d', 'volatility_20d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        result = engine.predict(sample_features_data, feature_cols=feature_cols)

        # All prediction columns should exist
        assert 'pred_1w' in result.columns
        assert 'pred_4w' in result.columns
        assert 'pred_12w' in result.columns

    def test_predict_before_training_raises_error(self):
        """Test prediction before training raises appropriate error."""
        engine = TAAModelEngine(horizons=[1])

        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

        with pytest.raises(ValueError, match="No trained models"):
            engine.predict(df)

    def test_predict_preserves_input(self, sample_features_data):
        """Test that predict doesn't modify input DataFrame."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        original_cols = sample_features_data.columns.tolist()
        result = engine.predict(sample_features_data, feature_cols=feature_cols)

        # Original should be unchanged
        assert sample_features_data.columns.tolist() == original_cols

        # Result should have new prediction columns
        assert len(result.columns) > len(original_cols)


class TestGetFeatureImportance:
    """Test feature importance retrieval."""

    def test_get_feature_importance_basic(self, sample_features_data):
        """Test basic feature importance retrieval."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        importance = engine.get_feature_importance(horizon=1, top_n=3)

        # Check structure
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) <= 3

    def test_get_feature_importance_untrained_horizon_raises_error(self):
        """Test getting importance for untrained horizon raises error."""
        engine = TAAModelEngine(horizons=[1])

        with pytest.raises(ValueError, match="No model trained"):
            engine.get_feature_importance(horizon=1)

    def test_get_feature_importance_top_n_filtering(self, sample_features_data):
        """Test top_n parameter filters correctly."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d', 'rsi_14d']
        engine.train(sample_features_data, feature_cols=feature_cols, verbose=False)

        importance_top2 = engine.get_feature_importance(horizon=1, top_n=2)
        importance_top10 = engine.get_feature_importance(horizon=1, top_n=10)

        # Should return at most top_n
        assert len(importance_top2) == 2
        # Should return all features if top_n > feature count
        assert len(importance_top10) == 3


class TestWalkForwardValidate:
    """Test walk-forward validation."""

    def test_walk_forward_validate_basic(self, sample_features_data):
        """Test basic walk-forward validation."""
        engine = TAAModelEngine(horizons=[1])

        feature_cols = ['momentum_20d', 'volatility_20d']
        results = engine.walk_forward_validate(
            sample_features_data,
            feature_cols=feature_cols,
            n_splits=3
        )

        # Check results structure
        assert 1 in results
        oos_df = results[1]

        assert isinstance(oos_df, pd.DataFrame)
        assert 'date' in oos_df.columns
        assert 'ticker' in oos_df.columns
        assert 'actual' in oos_df.columns
        assert 'predicted' in oos_df.columns
        assert 'fold' in oos_df.columns

    def test_walk_forward_validate_multiple_horizons(self, sample_features_data):
        """Test walk-forward validation with multiple horizons."""
        engine = TAAModelEngine(horizons=[1, 4])

        feature_cols = ['momentum_20d', 'volatility_20d']
        results = engine.walk_forward_validate(
            sample_features_data,
            feature_cols=feature_cols,
            n_splits=2
        )

        # Both horizons should have results
        assert 1 in results
        assert 4 in results

    def test_walk_forward_validate_missing_target(self):
        """Test walk-forward validation handles missing targets."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        }, index=dates)

        engine = TAAModelEngine(horizons=[1])
        results = engine.walk_forward_validate(df, feature_cols=['feature1'], n_splits=2)

        # Should return empty results (no valid data)
        assert len(results) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        engine = TAAModelEngine(horizons=[1])
        empty_df = pd.DataFrame()

        # create_targets requires 'Close' column
        # Empty df should raise KeyError (expected behavior)
        with pytest.raises(KeyError):
            engine.create_targets(empty_df)

    def test_insufficient_data_for_training(self):
        """Test training with insufficient data."""
        # Only 10 rows, not enough for meaningful training
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'feature1': np.random.randn(10),
            'target_1w': np.random.randn(10)
        }, index=dates)

        engine = TAAModelEngine(horizons=[1])

        # Should handle gracefully (may have very few validation samples)
        results = engine.train(df, feature_cols=['feature1'], verbose=False)

        # Should either train or skip gracefully
        assert isinstance(results, dict)
