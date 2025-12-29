"""
XGBoost Model Engine for TAA.
Trains multi-horizon models to forecast sector relative returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class TAAModelEngine:
    """
    XGBoost-based model engine for multi-horizon sector forecasting.
    
    Trains separate models for each prediction horizon (1w, 4w, 12w).
    Uses walk-forward validation to prevent look-ahead bias.
    
    Args:
        horizons: List of forward return horizons in weeks (e.g., [1, 4, 12]).
        params: XGBoost hyperparameters.
        
    Example:
        >>> engine = TAAModelEngine(horizons=[1, 4, 12])
        >>> engine.train(features_df, target_col='forward_return')
        >>> predictions = engine.predict(features_df)
    """
    
    def __init__(
        self,
        horizons: List[int] = [1, 4, 12],
        params: Optional[Dict] = None
    ):
        self.horizons = horizons
        self.params = params or self._default_params()
        self.models: Dict[int, xgb.XGBRegressor] = {}
        self.feature_importance_: Dict[int, pd.DataFrame] = {}
        
    def _default_params(self) -> Dict:
        """Default XGBoost parameters for TAA."""
        return {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'n_jobs': -1
        }
    
    def create_targets(
        self,
        data: pd.DataFrame,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        Create forward return targets for each horizon.
        
        Args:
            data: DataFrame with (Date, ticker) MultiIndex and price data.
            price_col: Name of price column to calculate returns from.
            
        Returns:
            pd.DataFrame: Original data with added target columns.
        """
        df = data.copy()
        
        # Calculate forward returns for each ticker
        for horizon in self.horizons:
            periods = horizon * 5  # Convert weeks to trading days
            target_col = f'target_{horizon}w'
            
            # Group by ticker and calculate forward returns
            if isinstance(df.index, pd.MultiIndex):
                # Assumes (Date, ticker) index
                df[target_col] = df.groupby(level='ticker')[price_col].pct_change(periods).shift(-periods)
            else:
                # Single ticker case
                df[target_col] = df[price_col].pct_change(periods).shift(-periods)
                
        return df
    
    def train(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        val_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[int, Dict]:
        """
        Train models for all horizons.
        
        Args:
            data: DataFrame with features and target columns.
            feature_cols: List of feature column names. If None, auto-detect.
            val_split: Fraction of data to use for validation.
            verbose: Whether to print training progress.
            
        Returns:
            Dict[int, Dict]: Training results for each horizon.
        """
        results = {}
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            target_cols = [f'target_{h}w' for h in self.horizons]
            exclude_cols = target_cols + ['Close', 'Open', 'High', 'Low', 'Volume']
            feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        if verbose:
            logger.info(f"Training with {len(feature_cols)} features: {feature_cols[:5]}...")
        
        for horizon in self.horizons:
            target_col = f'target_{horizon}w'
            
            if target_col not in data.columns:
                logger.warning(f"Target column {target_col} not found. Skipping {horizon}w model.")
                continue
            
            # Prepare train/val data (drop NaNs)
            valid_data = data[feature_cols + [target_col]].dropna()
            
            if len(valid_data) == 0:
                logger.warning(f"No valid data for {horizon}w horizon after dropping NaNs.")
                continue
            
            # Time-based split (last val_split of data)
            split_idx = int(len(valid_data) * (1 - val_split))
            
            X_train = valid_data.iloc[:split_idx][feature_cols]
            y_train = valid_data.iloc[:split_idx][target_col]
            X_val = valid_data.iloc[split_idx:][feature_cols]
            y_val = valid_data.iloc[split_idx:][target_col]
            
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {horizon}w horizon model")
                logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            
            # Train XGBoost model
            model = xgb.XGBRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=verbose
            )
            
            # Store model
            self.models[horizon] = model
            
            # Calculate metrics
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_ic = np.corrcoef(y_train, train_pred)[0, 1]
            val_ic = np.corrcoef(y_val, val_pred)[0, 1]
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance_[horizon] = importance_df
            
            results[horizon] = {
                'train_ic': train_ic,
                'val_ic': val_ic,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'top_features': importance_df.head(10)['feature'].tolist()
            }
            
            if verbose:
                logger.info(f"Train IC: {train_ic:.4f}, Val IC: {val_ic:.4f}")
                logger.info(f"Top 5 features: {results[horizon]['top_features'][:5]}")
        
        return results
    
    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for all horizons.
        
        Args:
            data: DataFrame with features.
            feature_cols: List of feature column names. If None, use training features.
            
        Returns:
            pd.DataFrame: Original data with added prediction columns.
        """
        df = data.copy()
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Use first model's feature names
            if self.models:
                first_model = list(self.models.values())[0]
                feature_cols = first_model.get_booster().feature_names
            else:
                raise ValueError("No trained models found. Call train() first.")
        
        for horizon, model in self.models.items():
            pred_col = f'pred_{horizon}w'
            
            # Predict only on rows with valid features
            valid_mask = df[feature_cols].notna().all(axis=1)
            valid_data = df.loc[valid_mask, feature_cols]
            
            if len(valid_data) > 0:
                df.loc[valid_mask, pred_col] = model.predict(valid_data)
            
        return df
    
    def get_feature_importance(
        self,
        horizon: int,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance for a specific horizon.
        
        Args:
            horizon: Forecast horizon in weeks.
            top_n: Number of top features to return.
            
        Returns:
            pd.DataFrame: Feature importance ranking.
        """
        if horizon not in self.feature_importance_:
            raise ValueError(f"No model trained for {horizon}w horizon.")
        
        return self.feature_importance_[horizon].head(top_n)
    
    def walk_forward_validate(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        n_splits: int = 5,
        expanding_window: bool = True
    ) -> Dict[int, pd.DataFrame]:
        """
        Perform walk-forward validation.
        
        Args:
            data: DataFrame with features and targets.
            feature_cols: List of feature column names.
            n_splits: Number of train/test splits.
            expanding_window: If True, use expanding window. Else rolling window.
            
        Returns:
            Dict[int, pd.DataFrame]: Out-of-sample predictions for each horizon.
        """
        results = {}
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for horizon in self.horizons:
            target_col = f'target_{horizon}w'
            
            if target_col not in data.columns:
                logger.warning(f"Target {target_col} not found. Skipping.")
                continue
            
            # Prepare data
            valid_data = data[feature_cols + [target_col]].dropna()
            X = valid_data[feature_cols]
            y = valid_data[target_col]
            
            oos_predictions = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                logger.info(f"Fold {fold+1}/{n_splits} for {horizon}w horizon")
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model = xgb.XGBRegressor(**self.params)
                model.fit(X_train, y_train, verbose=False)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Store predictions
                oos_df = pd.DataFrame({
                    'date': X_test.index.get_level_values('Date') if isinstance(X_test.index, pd.MultiIndex) else X_test.index,
                    'ticker': X_test.index.get_level_values('ticker') if isinstance(X_test.index, pd.MultiIndex) else 'unknown',
                    'actual': y_test.values,
                    'predicted': y_pred,
                    'fold': fold
                })
                oos_predictions.append(oos_df)
            
            results[horizon] = pd.concat(oos_predictions, ignore_index=True)
        
        return results
