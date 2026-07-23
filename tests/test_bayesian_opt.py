import pytest
import numpy as np
import pandas as pd
from models.xgb_model import XGBoostTrainer
from models.random_forest import RandomForestTrainer

@pytest.fixture
def synthetic_classification_data():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def test_optuna_xgboost_tuning(synthetic_classification_data):
    X, y = synthetic_classification_data
    trainer = XGBoostTrainer(n_splits=3, purge_size=5, embargo_size=2)
    
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    best_params = trainer.find_best_params(X, y, param_dist, n_iter=3, use_bayesian=True)
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params
    assert 'max_depth' in best_params

def test_optuna_random_forest_tuning(synthetic_classification_data):
    X, y = synthetic_classification_data
    trainer = RandomForestTrainer(n_splits=3, purge_size=5, embargo_size=2)
    
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']
    }
    
    best_params = trainer.find_best_params(X, y, param_dist, n_iter=3, use_bayesian=True)
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params
    assert 'max_depth' in best_params

