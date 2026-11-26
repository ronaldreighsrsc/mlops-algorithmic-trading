import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
import mlflow

# Agregar la carpeta src al PYTHONPATH
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class CPCVAuditor:
    """
    Auditor Estadístico basado en Combinatorial Purged Cross-Validation (CPCV).
    Basado en la arquitectura matemática de Marcos López de Prado.
    Calcula la Probabilidad de Overfitting del Backtest (PBO) mediante
    la evaluación de N combinaciones de caminos cruzados.
    """
    def __init__(self, symbol: str = "EURUSD", results_dir: str = None):
        self.symbol = symbol
        self.base_dir = base_dir
        self.results_dir = results_dir or os.path.join(os.path.dirname(base_dir), "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_combinatorial_splits(self, n_samples: int, n_groups: int = 6, k_test: int = 2, purge_size: int = 10):
        """
        Genera todas las combinaciones posibles C(n_groups, k_test) de caminos de prueba.
        Aplica Purging en las fronteras para evitar fuga de datos.
        """
        group_size = n_samples // n_groups
        groups = []
        for i in range(n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < n_groups - 1 else n_samples
            groups.append((start, end))

        combos = list(itertools.combinations(range(n_groups), k_test))
        paths = []

        for test_group_indices in combos:
            test_indices = []
            for g_idx in test_group_indices:
                g_start, g_end = groups[g_idx]
                test_indices.extend(list(range(g_start, g_end)))

            test_indices = np.array(test_indices)
            
            # El resto de grupos forman el tren
            train_group_indices = [g for g in range(n_groups) if g not in test_group_indices]
            train_indices = []
            for g_idx in train_group_indices:
                g_start, g_end = groups[g_idx]
                # Aplicar purging en los bordes adyacentes a bloques de prueba
                purged_start = g_start
                purged_end = g_end
                for t_idx in test_group_indices:
                    if t_idx == g_idx - 1:
                        purged_start += purge_size # Purge al inicio si la prueba precede
                    elif t_idx == g_idx + 1:
                        purged_end -= purge_size # Purge al final si la prueba sucede
                
                if purged_start < purged_end:
                    train_indices.extend(list(range(purged_start, purged_end)))

            train_indices = np.array(train_indices)
            paths.append((train_indices, test_indices))

        return paths

    def calculate_path_sharpe(self, returns: np.ndarray, rf: float = 0.0) -> float:
        """Calcula el Sharpe Ratio de un camino específico."""
        if len(returns) < 5:
            return 0.0
        std = np.std(returns)
        if std == 0 or np.isnan(std):
            return 0.0
        mean_ret = np.mean(returns) - rf
        sharpe = (mean_ret / std) * np.sqrt(252) # Anualizado
        return float(sharpe)

    def audit_strategy(self, df_processed: pd.DataFrame, probs: np.ndarray, confidence_threshold: float = 0.50, n_groups: int = 6, k_test: int = 2) -> dict:
        """
        Ejecuta la auditoría CPCV completa sobre la serie de predicciones y retornos.
        Retorna la métrica PBO y la distribución de Sharpe Ratios.
        """
        n_samples = min(len(df_processed), len(probs))
        if n_samples < 100:
            print(f"⚠️ Insuficientes muestras para CPCV en {self.symbol} (Muestras: {n_samples}).")
            return {"pbo_pct": 0.0, "sharpe_mean": 0.0, "sharpe_std": 0.0, "n_paths": 0}

        df_sub = df_processed.iloc[-n_samples:].copy()
        probs_sub = probs[-n_samples:]
        
        # Retorno diario simulado de la estrategia (ESTRICTO shift(1): la señal del cierre t-1 opera el retorno de t)
        close_prices = df_sub['close'].values
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = np.insert(daily_returns, 0, 0.0)

        raw_signals = (probs_sub > confidence_threshold).astype(int)
        signals = np.insert(raw_signals[:-1], 0, 0)
        strategy_returns = daily_returns * signals

        paths = self.generate_combinatorial_splits(n_samples, n_groups=n_groups, k_test=k_test)
        sharpe_list = []

        for train_idx, test_idx in paths:
            path_returns = strategy_returns[test_idx]
            sharpe = self.calculate_path_sharpe(path_returns)
            sharpe_list.append(sharpe)

        sharpes = np.array(sharpe_list)
        # PBO es el % de caminos donde el Sharpe Ratio resulta <= 0 (Degradación/Overfitting)
        pbo_count = np.sum(sharpes <= 0.0)
        pbo_pct = float(pbo_count / len(sharpes)) if len(sharpes) > 0 else 1.0

        sharpe_mean = float(np.mean(sharpes))
        sharpe_std = float(np.std(sharpes))
        sharpe_min = float(np.min(sharpes))
        sharpe_max = float(np.max(sharpes))

        print(f"\n==================================================")
        print(f"🔬 AUDITORÍA CPCV & PBO - {self.symbol}")
        print(f"==================================================")
        print(f"  > Combinaciones de Caminos (Paths): {len(sharpes)}")
        print(f"  > Sharpe Medio (CPCV): {sharpe_mean:.2f} (±{sharpe_std:.2f})")
        print(f"  > Rango de Sharpe: [{sharpe_min:.2f}, {sharpe_max:.2f}]")
        print(f"  > PBO (Probability of Backtest Overfitting): {pbo_pct:.2%}")
        
        if pbo_pct < 0.05:
            print(f"  ✅ ESTRATEGIA CERTIFICADA: PBO < 5% (Excelente Robustez Estadística).")
        elif pbo_pct < 0.20:
            print(f"  ⚠️ ROBUSTEZ MODERADA: PBO < 20% (Aceptable en mercados volátiles).")
        else:
            print(f"  ❌ RIESGO DE OVERFITTING ALTO: PBO >= 20%.")
        print(f"==================================================\n")

        # Generar Gráfico de Distribución de Sharpe
        plot_path = self._plot_sharpe_distribution(sharpes, pbo_pct)

        # Logging a MLflow
        try:
            mlflow.set_experiment("CPCV_PBO_Audits")
            with mlflow.start_run(run_name=f"CPCV_Audit_{self.symbol}"):
                mlflow.log_metrics({
                    "PBO_Pct": pbo_pct,
                    "Sharpe_Mean": sharpe_mean,
                    "Sharpe_Std": sharpe_std,
                    "Sharpe_Min": sharpe_min,
                    "Sharpe_Max": sharpe_max,
                    "Paths_Count": float(len(sharpes))
                })
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="cpcv_plots")
        except Exception as ml_err:
            print(f"⚠️ MLflow logging diferido: {ml_err}")

        return {
            "symbol": self.symbol,
            "pbo_pct": pbo_pct,
            "sharpe_mean": sharpe_mean,
            "sharpe_std": sharpe_std,
            "sharpe_min": sharpe_min,
            "sharpe_max": sharpe_max,
            "n_paths": len(sharpes),
            "plot_path": plot_path
        }

    def _plot_sharpe_distribution(self, sharpes: np.ndarray, pbo_pct: float) -> str:
        """Genera e imprime el histograma de distribución de Sharpe Ratios CPCV."""
        plt.figure(figsize=(9, 5))
        plt.hist(sharpes, bins=12, color='#2b5c8f', edgecolor='black', alpha=0.75, label='Caminos CPCV')
        plt.axvline(0.0, color='red', linestyle='--', linewidth=2, label=f'Umbral Cero (PBO: {pbo_pct:.1%})')
        plt.axvline(np.mean(sharpes), color='green', linestyle='-', linewidth=2, label=f'Sharpe Medio ({np.mean(sharpes):.2f})')
        
        plt.title(f"Distribución de Sharpe Ratios CPCV - {self.symbol}\n(Combinatorial Purged Cross-Validation)", fontsize=13, fontweight='bold')
        plt.xlabel("Sharpe Ratio Anualizado", fontsize=11)
        plt.ylabel("Frecuencia de Caminos", fontsize=11)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.results_dir, f"cpcv_sharpe_distribution_{self.symbol}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    @staticmethod
    def compute_deflated_sharpe_ratio(
        returns: np.ndarray,
        n_trials: int = 1,
        trial_sharpes: np.ndarray | None = None,
        annualization: float = 252.0,
    ) -> dict:
        """
        Deflated Sharpe Ratio (DSR) formal de Bailey & López de Prado (2014).
        Usa la varianza EMPÍRICA real de los Sharpe Ratios del torneo (sin hardcodeo 0.5).
        """
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 5:
            return {"sharpe_obs": 0.0, "sr0_null": 0.0, "dsr": 0.0, "psr": 0.0, "skew": 0.0, "kurtosis": 3.0}

        std = float(np.std(r, ddof=1))
        if std <= 1e-12:
            return {"sharpe_obs": 0.0, "sr0_null": 0.0, "dsr": 0.0, "psr": 0.0, "skew": 0.0, "kurtosis": 3.0}

        sr_non_ann = float(np.mean(r) / std)
        sr_ann = sr_non_ann * float(np.sqrt(annualization))
        sk = float(skew(r))
        # Curtosis total (normal = 3.0) como exige eq. (5) de Bailey & López de Prado
        ku = float(kurtosis(r, fisher=False))

        # Varianza del estimador de Sharpe por observación
        sr_var = (1.0 - sk * sr_non_ann + ((ku - 1.0) / 4.0) * (sr_non_ann ** 2)) / max(1, n - 1)
        sigma_sr = float(np.sqrt(max(sr_var, 1e-12)))

        psr = float(norm.cdf(sr_non_ann / sigma_sr))

        # Umbral nulo por Multiple Testing (N ensayos)
        eff_n = max(1, int(n_trials))
        if eff_n <= 1:
            sr0_non_ann = 0.0
        else:
            euler_gamma = 0.5772156649
            z_max = (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / eff_n) + euler_gamma * norm.ppf(
                1.0 - (1.0 / eff_n) * np.exp(-1.0)
            )
            if trial_sharpes is not None and len(trial_sharpes) >= 2:
                ts_arr = np.asarray(trial_sharpes, dtype=np.float64)
                ts_arr = ts_arr[np.isfinite(ts_arr)]
                std_trials = float(np.std(ts_arr, ddof=1)) if len(ts_arr) >= 2 else sigma_sr
            else:
                std_trials = sigma_sr
            sr0_non_ann = std_trials * float(z_max)

        dsr = float(norm.cdf((sr_non_ann - sr0_non_ann) / sigma_sr))
        return {
            "sharpe_obs": sr_ann,
            "sr_non_ann": sr_non_ann,
            "sr0_null": sr0_non_ann * float(np.sqrt(annualization)),
            "dsr": dsr,
            "psr": psr,
            "skew": sk,
            "kurtosis": ku,
        }

    def audit_cscv_pbo(
        self,
        strategy_returns_matrix: np.ndarray | pd.DataFrame,
        n_blocks: int = 10,
        purge_bars: int = 10,
        max_combinations: int = 252,
    ) -> dict:
        """
        Implementación formal del algoritmo CSCV (Combinatorially Symmetric Cross-Validation)
        para calcular la Probabilidad de Sobreajuste de Backtest (PBO — Bailey et al. 2014).

        :param strategy_returns_matrix: Matriz T x N donde cada columna es la serie de
                                        retornos netos de una estrategia candidata del torneo.
        :param n_blocks: Número par S de bloques temporales contiguos (ej. 8 o 10).
        :param purge_bars: Barras purgadas en las fronteras entre bloques IS y OOS.
        """
        M = np.asarray(strategy_returns_matrix, dtype=np.float64)
        if M.ndim != 2:
            raise ValueError("strategy_returns_matrix debe ser una matriz 2D (T x N).")

        T, N = M.shape
        if N < 2:
            raise ValueError("El PBO formal CSCV requiere al menos N >= 2 estrategias competidoras.")
        if n_blocks % 2 != 0 or n_blocks < 4:
            raise ValueError("n_blocks debe ser un entero par >= 4.")
        if T < n_blocks * 10:
            return {
                "pbo": 1.0,
                "median_logit": -1.0,
                "prob_oos_loss": 1.0,
                "mean_is_sharpe": 0.0,
                "mean_oos_sharpe": 0.0,
                "degradation_slope": 0.0,
                "n_combinations": 0,
                "n_strategies": N,
            }

        block_size = T // n_blocks
        blocks = [(b * block_size, (b + 1) * block_size if b < n_blocks - 1 else T) for b in range(n_blocks)]

        all_combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
        if len(all_combos) > max_combinations:
            rng = np.random.default_rng(42)
            sel = rng.choice(len(all_combos), size=max_combinations, replace=False)
            combos = [all_combos[idx] for idx in sorted(sel)]
        else:
            combos = all_combos

        logits = []
        is_best_sharpes = []
        oos_selected_sharpes = []

        for is_block_ids in combos:
            is_set = set(is_block_ids)
            oos_block_ids = [b for b in range(n_blocks) if b not in is_set]

            is_idx = []
            for b in is_block_ids:
                s, e = blocks[b]
                if (b - 1) in oos_block_ids:
                    s += purge_bars
                if (b + 1) in oos_block_ids:
                    e -= purge_bars
                if s < e:
                    is_idx.extend(range(s, e))

            oos_idx = []
            for b in oos_block_ids:
                s, e = blocks[b]
                oos_idx.extend(range(s, e))

            if len(is_idx) < 10 or len(oos_idx) < 10:
                continue

            M_is = M[is_idx, :]
            M_oos = M[oos_idx, :]

            # Sharpe por columna en IS y OOS (sin warnings si std == 0)
            is_std = np.std(M_is, axis=0, ddof=1)
            is_mean = np.mean(M_is, axis=0)
            is_sr = np.full(N, -1e9, dtype=np.float64)
            np.divide(is_mean * np.sqrt(252.0), is_std, out=is_sr, where=is_std > 1e-12)

            oos_std = np.std(M_oos, axis=0, ddof=1)
            oos_mean = np.mean(M_oos, axis=0)
            oos_sr = np.zeros(N, dtype=np.float64)
            np.divide(oos_mean * np.sqrt(252.0), oos_std, out=oos_sr, where=oos_std > 1e-12)

            n_star = int(np.argmax(is_sr))
            sr_is_star = float(is_sr[n_star])
            sr_oos_star = float(oos_sr[n_star])

            # Rango relativo en OOS: rank ∈ [1, N] -> omega_bar = rank / (N + 1) ∈ (0, 1)
            # Empates resueltos por rango medio (rankdata)
            oos_ranks = pd.Series(oos_sr).rank(method="average").values
            omega_bar = float(np.clip(oos_ranks[n_star] / (N + 1.0), 1e-6, 1.0 - 1e-6))
            lam = float(np.log(omega_bar / (1.0 - omega_bar)))

            logits.append(lam)
            is_best_sharpes.append(sr_is_star)
            oos_selected_sharpes.append(sr_oos_star)

        logits_arr = np.asarray(logits, dtype=np.float64)
        is_arr = np.asarray(is_best_sharpes, dtype=np.float64)
        oos_arr = np.asarray(oos_selected_sharpes, dtype=np.float64)

        pbo = float(np.mean(logits_arr < 0.0)) if len(logits_arr) > 0 else 1.0
        prob_oos_loss = float(np.mean(oos_arr <= 0.0)) if len(oos_arr) > 0 else 1.0
        slope = 0.0
        if len(is_arr) >= 2 and np.std(is_arr) > 1e-12:
            slope = float(np.polyfit(is_arr, oos_arr, 1)[0])

        return {
            "pbo": pbo,
            "median_logit": float(np.median(logits_arr)) if len(logits_arr) > 0 else 0.0,
            "prob_oos_loss": prob_oos_loss,
            "mean_is_sharpe": float(np.mean(is_arr)) if len(is_arr) > 0 else 0.0,
            "mean_oos_sharpe": float(np.mean(oos_arr)) if len(oos_arr) > 0 else 0.0,
            "degradation_slope": slope,
            "n_combinations": len(logits_arr),
            "n_strategies": N,
        }


if __name__ == "__main__":
    import json
    activos = ["EURUSD", "EURUSD_H4", "SP500", "SP500_H4", "Oro", "Oro_H4"]
    
    for symbol in activos:
        proc_path = os.path.join(base_dir, "data", "processed", f"{symbol}_processed.csv")
        probs_file = [f for f in os.listdir(os.path.join(base_dir, "results")) if f.startswith("probs_") and symbol in f]
        
        if os.path.exists(proc_path) and probs_file:
            print(f"\n🔍 Auditoría CPCV & PBO para {symbol}...")
            df_proc = pd.read_csv(proc_path)
            probs_path = os.path.join(base_dir, "results", probs_file[0])
            probs = np.load(probs_path)
            
            auditor = CPCVAuditor(symbol=symbol)
            auditor.audit_strategy(df_proc, probs)

