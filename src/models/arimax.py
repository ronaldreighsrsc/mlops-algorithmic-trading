import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import warnings

class ARIMAXTrainer:
    """
    Motor estadístico responsable de entrenar, buscar hiperparámetros y 
    generar predicciones estep-by-step (Walk-Forward) usando ARIMAX.
    """
    def __init__(self, p_values: list = [0, 1, 2, 5, 10], q_values: list = [0, 1, 2], retrain_step: int = 50):
        self.p_values = p_values
        self.q_values = q_values
        self.retrain_step = retrain_step

    def find_best_order(self, train_target: pd.Series, train_exog: pd.DataFrame = None) -> tuple:
        """
        Realiza un Grid Search direccional sobre el set de entrenamiento
        (split 80/20 interno) para encontrar el mejor orden (p, 0, q).
        """
        print(f"  🔍 Buscando hiperparámetros óptimos (Grid Search Direccional)...")
        inner_split = int(len(train_target) * 0.8)
        
        it_target = train_target.iloc[:inner_split]
        iv_target = train_target.iloc[inner_split:]
        
        it_exog = train_exog.iloc[:inner_split] if train_exog is not None and not train_exog.empty else None
        iv_exog = train_exog.iloc[inner_split:] if train_exog is not None and not train_exog.empty else None

        # Lógica para Direccionalidad (Up/Down)
        prev_iv = np.roll(iv_target.values, 1)
        prev_iv[0] = it_target.iloc[-1]
        y_iv_bin = (iv_target.values > prev_iv).astype(int)

        best_acc, best_order = -1.0, (1, 0, 1) # Default de seguridad

        for p in self.p_values:
            for q in self.q_values:
                try:
                    # Atrapamos warnings matemáticos internos para no ensuciar la consola
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(it_target, exog=it_exog, order=(p, 0, q)).fit()
                        forecasts = model.forecast(steps=len(iv_target), exog=iv_exog)
                        
                        # Si el modelo explota y devuelve NaNs, lo descartamos
                        if forecasts.isna().any(): continue
                        
                        acc = accuracy_score(y_iv_bin, (forecasts.values > prev_iv).astype(int))
                        if acc > best_acc:
                            best_acc = acc
                            best_order = (p, 0, q)
                except Exception:
                    continue # Falla de convergencia de álgebra lineal, saltar intento

        print(f"  ✅ Mejor orden: {best_order} (Inner Acc: {best_acc:.2%})")
        return best_order

    def walk_forward_predict(self, target: pd.Series, exog: pd.DataFrame, train_size: int, best_order: tuple) -> np.ndarray:
        """
        Ejecuta la validación Walk-Forward, reentrenando el modelo cada X pasos
        y devolviendo la probabilidad de que el precio suba al día siguiente.
        """
        print(f"  🚀 Iniciando Walk-Forward (Reentrenamiento cada {self.retrain_step} días)...")
        
        train_target = target.iloc[:train_size]
        test_target = target.iloc[train_size:]
        
        train_exog = exog.iloc[:train_size] if exog is not None and not exog.empty else None
        test_exog = exog.iloc[train_size:] if exog is not None and not exog.empty else None

        pred_probs = []

        # 1. Entrenamiento del modelo base
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curr_model = ARIMA(train_target, exog=train_exog, order=best_order).fit()

        # 2. Iteración paso a paso sobre el Test Set
        for i in range(len(test_target)):
            c_exog = test_exog.iloc[[i]] if test_exog is not None and not test_exog.empty else None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = curr_model.get_forecast(steps=1, exog=c_exog)
            
            # Valor del día anterior para calcular si "sube" o "baja"
            p_val = train_target.iloc[-1] if i == 0 else test_target.iloc[i-1]
            
            mean = f.predicted_mean.iloc[0]
            se = f.se_mean.iloc[0]
            
            # Fail-Fast: Si la varianza colapsa, evitamos la división por cero
            if pd.isna(se) or se <= 0: 
                se = 1e-6
                
            # Mapeo a Probabilidad usando la CDF
            prob = 1.0 - norm.cdf(p_val, loc=mean, scale=se)
            pred_probs.append(prob)
            
            if (i + 1) % self.retrain_step == 0:
                t_t_upd = target.iloc[:train_size + i + 1]
                t_e_upd = exog.iloc[:train_size + i + 1] if exog is not None and not exog.empty else None
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    curr_model = ARIMA(t_t_upd, exog=t_e_upd, order=best_order).fit()

        self.curr_model = curr_model
        self.best_order = best_order
        return np.array(pred_probs)

    def save(self, filepath: str) -> None:
        """Saves the final trained model."""
        if not hasattr(self, 'curr_model'):
            raise ValueError("No model trained yet. Run walk_forward_predict first.")
            
        import joblib
        state = {
            'model': self.curr_model,
            'p_values': self.p_values,
            'q_values': self.q_values,
            'retrain_step': self.retrain_step,
            'best_order': getattr(self, 'best_order', None)
        }
        joblib.dump(state, filepath)
        print(f"  💾 ARIMAX model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved model."""
        import joblib
        state = joblib.load(filepath)
        
        instance = cls(
            p_values=state['p_values'],
            q_values=state['q_values'],
            retrain_step=state['retrain_step']
        )
        instance.curr_model = state['model']
        instance.best_order = state.get('best_order', None)
        return instance
        
    def fast_retrain(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'close_FFD'):
        """Reentrena el filtro ARIMA sobre los datos más recientes."""
        if not hasattr(self, 'best_order') or self.best_order is None:
            print("  ⚠️ No best_order found. Cannot fast retrain.")
            return
            
        df_valid = df.dropna(subset=[target_col])
        if len(df_valid) == 0:
            return
            
        train_target = df_valid[target_col].iloc[-1000:]
        train_exog = df_valid[feature_cols].iloc[-1000:] if feature_cols else None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.curr_model = ARIMA(train_target, exog=train_exog, order=self.best_order).fit()
            
        print("  ✅ Pesos de ARIMAX reentrenados exitosamente con datos recientes.")