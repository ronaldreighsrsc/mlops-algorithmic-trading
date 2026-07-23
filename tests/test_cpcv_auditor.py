import pytest
import numpy as np
import pandas as pd
from evaluation.cpcv_auditor import CPCVAuditor

@pytest.fixture
def synthetic_cpcv_data():
    np.random.seed(42)
    n = 300
    prices = 1.0 + np.cumsum(np.random.randn(n) * 0.005)
    df_proc = pd.DataFrame({'close': prices})
    probs = np.random.uniform(0.4, 0.7, n)
    return df_proc, probs

def test_generate_combinatorial_splits_count():
    auditor = CPCVAuditor(symbol="TEST")
    paths = auditor.generate_combinatorial_splits(n_samples=600, n_groups=6, k_test=2)
    assert len(paths) == 15 # 6C2 = 15
    for train_idx, test_idx in paths:
        assert len(train_idx) > 0
        assert len(test_idx) > 0

def test_calculate_path_sharpe():
    auditor = CPCVAuditor(symbol="TEST")
    # Retornos positivos con varianza positiva
    returns = np.array([0.01, 0.02, 0.015, -0.005, 0.01])
    sharpe = auditor.calculate_path_sharpe(returns)
    assert sharpe > 0

    # Retornos planos (varianza cero)
    flat_returns = np.array([0.0, 0.0, 0.0, 0.0])
    assert auditor.calculate_path_sharpe(flat_returns) == 0.0

def test_audit_strategy_pbo_metric(synthetic_cpcv_data):
    df_proc, probs = synthetic_cpcv_data
    auditor = CPCVAuditor(symbol="TEST_PBO")
    results = auditor.audit_strategy(df_proc, probs, n_groups=6, k_test=2)

    assert "pbo_pct" in results
    assert "sharpe_mean" in results
    assert 0.0 <= results["pbo_pct"] <= 1.0
    assert results["n_paths"] == 15
    assert results["plot_path"] is not None

def test_insufficient_samples():
    auditor = CPCVAuditor(symbol="TEST_SHORT")
    df_short = pd.DataFrame({'close': [1.0, 1.1, 1.2]})
    probs_short = np.array([0.5, 0.6, 0.7])
    results = auditor.audit_strategy(df_short, probs_short)
    assert results["pbo_pct"] == 0.0
    assert results["n_paths"] == 0
