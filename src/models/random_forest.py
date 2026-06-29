import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class RandomForestTrainer:
    """
    Motor de Machine Learning utilizando Random Forest.
    Implementa preparación de secuencias (Look-Back), escalamiento,
    validación cruzada avanzada (Purged & Embargoed) y predicción Walk-Forward.
    """
    def __init__(self, look_back: int = 60, retrain_step: int = 50, 
                 n_splits: int = 3, purge_size: int = 60, embargo_size: int = 10):
        self.look_back = look_back
        self.retrain_step = retrain_step
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.scaler = StandardScaler()

    def prepare_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
        """
        Transforma el DataFrame bidimensional en secuencias planas (ventanas de tiempo) 
        para que Random Forest pueda procesar memoria histórica.
        """
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        y_array = df_clean[target_col].values
        features = [c for c in feature_cols if c != target_col]
        dataset = df_clean[features].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            window = dataset[i-self.look_back:i, :]
            X_raw.append(window.flatten())
            y_raw.append(y_array[i])
            
        return np.array(X_raw), np.array(y_raw)

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Genera los índices para validación cruzada evitando Data Leakage."""
        fold_size = num_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else num_samples
            train_indices = []
            for j in range(num_samples):
                if j < val_start - self.purge_size:          
                    train_indices.append(j)
                elif j >= val_end + self.embargo_size:       
                    train_indices.append(j)
            val_indices = list(range(val_start, val_end))
            folds.append((np.array(train_indices), np.array(val_indices)))
        return folds

    def find_best_params(self, X_train: np.ndarray, y_train: np.ndarray, param_distributions: dict, n_iter: int = 10) -> dict:
        """Ejecuta RandomizedSearchCV usando Purged & Embargoed CV sobre los datos de entrenamiento."""
        from sklearn.model_selection import RandomizedSearchCV
        print(f"  🔍 Buscando hiperparámetros (RandomizedSearchCV - Purged & Embargoed CV)...")
        folds = self._get_purged_embargoed_folds(len(X_train))
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=folds,
            scoring='f1_macro', # Optimizar para predecir bien ambas clases
            random_state=42,
            n_jobs=1 # Para no sobrecargar si n_jobs del RF ya es -1
        )
        
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"  ✅ Ganador: {best_params} (Score CV: {best_score:.2f})")
        return best_params

    def walk_forward_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray, best_params: dict) -> tuple:
        """
        Escala los datos sin Leakage, entrena y predice paso a paso, 
        reentrenando periódicamente. Retorna probabilidades e importancia de variables.
        """
        print(f"  🚀 Iniciando Walk-Forward (Reentrenamiento cada {self.retrain_step} días)...")
        
        # 1. Escalamiento estricto (Fit solo en train, transform en test)
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -10, 10)
        X_test_scaled = np.clip(self.scaler.transform(X_test), -10, 10)

        # 2. Modelo Base
        master_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        master_model.fit(X_train_scaled, y_train)

        pred_probs = []
        
        # 3. Iteración Paso a Paso
        for i in range(len(X_test_scaled)):
            prob = master_model.predict_proba(X_test_scaled[i].reshape(1, -1))[0][1]
            pred_probs.append(prob)
            
            # Reentrenamiento periódico
            if (i + 1) % self.retrain_step == 0:
                curr_X = np.concatenate((X_train_scaled, X_test_scaled[:i+1]))
                curr_y = np.concatenate((y_train, y_test[:i+1]))
                master_model.fit(curr_X, curr_y)

        # 4. Feature Importance final (del último modelo ajustado)
        # Calculamos cuántas variables originales hay (columnas totales / días de look_back)
        num_features = X_train.shape[1] // self.look_back
        importances_flat = master_model.feature_importances_
        
        # Agrupamos la importancia de los N días hacia atrás para cada variable original
        importances = importances_flat.reshape(self.look_back, num_features).sum(axis=0)
        
        self.master_model = master_model
        self.best_params = best_params
        return np.array(pred_probs), importances

    def save(self, filepath: str) -> None:
        """Saves the final trained model, scaler and params."""
        if not hasattr(self, 'master_model'):
            raise ValueError("No model trained yet. Run walk_forward_predict first.")
        state = {
            'scaler': self.scaler,
            'look_back': self.look_back,
            'master_model': self.master_model,
            'best_params': getattr(self, 'best_params', None)
        }
        joblib.dump(state, filepath)
        print(f"  💾 Random Forest model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved model."""
        state = joblib.load(filepath)
        instance = cls(look_back=state['look_back'])
        instance.scaler = state['scaler']
        instance.master_model = state['master_model']
        instance.best_params = state.get('best_params', None)
        return instance
        
    def fast_retrain(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'Label'):
        """Reentrena los pesos del modelo usando los datos más recientes y los best_params históricos."""
        if not hasattr(self, 'best_params') or self.best_params is None:
            print("  ⚠️ No best_params found. Cannot fast retrain.")
            return
            
        if target_col not in df.columns:
            # En producción live no calculamos Label (requiere ver el futuro)
            return
            
        df_valid = df.dropna(subset=[target_col])
        if len(df_valid) == 0:
            return
            
        X, y = self.prepare_sequences(df_valid, target_col, feature_cols)
        
        X_train = X[-1000:]
        y_train = y[-1000:]
        
        X_train_scaled = np.clip(self.scaler.transform(X_train), -10, 10)
        
        self.master_model = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
        self.master_model.fit(X_train_scaled, y_train)
        print("  ✅ Pesos de Random Forest reentrenados exitosamente con datos recientes.")