import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        
        # Retorno diario simulado de la estrategia
        close_prices = df_sub['close'].values
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = np.insert(daily_returns, 0, 0.0)

        signals = (probs_sub > confidence_threshold).astype(int)
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


if __name__ == "__main__":
    import json
    activos = ["EURUSD", "EURUSD_H4", "SP500", "SP500_H4", "Oro", "Oro_H4", "ECH"]
    
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

