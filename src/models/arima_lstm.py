import os
import warnings

# Silenciar logs de C++ de TensorFlow ANTES de importarlo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

class HybridARIMALSTMTrainer:
    """
    Motor Híbrido (Estadística + Deep Learning).
    Fase 1: Ajusta un filtro lineal (ARIMA) para capturar la estructura media.
    Fase 2: Extrae los residuos y los combina con variables exógenas.
    Fase 3: Entrena un LSTM Clasificador sobre los residuos para predecir la dirección.
    """
    def __init__(self, look_back: int = 60, retrain_step: int = 50, 
                 n_splits: int = 3, purge_size: int = 60, embargo_size: int = 10,
                 p_values: list = [0, 1, 2, 5, 10], q_values: list = [0, 1, 2]):
        self.look_back = look_back
        self.retrain_step = retrain_step
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.p_values = p_values
        self.q_values = q_values

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Genera índices para validación cruzada evitando Data Leakage."""
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

    def _build_lstm_classifier(self, input_shape: tuple, units: int, dropout: float) -> Sequential:
        """Red neuronal para aprender de los errores del ARIMA."""
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _create_3d_sequences(self, residuals: pd.Series, exog: pd.DataFrame, target: pd.Series) -> tuple:
        """Combina residuos y exógenas en tensores 3D para el LSTM."""
        if exog is not None and not exog.empty:
            df_merged = pd.concat([residuals.rename('Residual'), exog], axis=1).dropna()
        else:
            df_merged = residuals.rename('Residual').to_frame().dropna()
            
        dataset_vals = df_merged.values
        target_vals = target.loc[df_merged.index].values
        n_feat = dataset_vals.shape[1]

        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset_vals)):
            X_raw.append(dataset_vals[i-self.look_back:i, :])
            y_raw.append(1 if target_vals[i] > target_vals[i-1] else 0)

        return np.array(X_raw), np.array(y_raw), n_feat

    def find_best_params(self, train_target: pd.Series, train_exog: pd.DataFrame, param_distributions: dict, n_iter: int = 3, use_bayesian: bool = True) -> dict:

        """
        1. Búsqueda por AIC para el mejor ARIMA.
        2. Extracción de residuos.
        3. Optimización Bayesiana (Optuna) o ParameterSampler para el LSTM Clasificador.
        """
        print(f"  🔍 [FASE 1] Buscando filtro lineal óptimo (Minimizando AIC)...")
        best_aic = np.inf
        best_arima_order = (1, 0, 1)
        
        # Búsqueda ARIMA
        for p in self.p_values:
            for q in self.q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        temp_model = ARIMA(train_target, order=(p, 0, q)).fit()
                        if temp_model.aic < best_aic:
                            best_aic = temp_model.aic
                            best_arima_order = (p, 0, q)
                except Exception:
                    continue
        print(f"    > Ganador ARIMA: {best_arima_order} (AIC: {best_aic:.2f})")

        # Preparar datos LSTM
        arima_model = ARIMA(train_target, order=best_arima_order).fit()
        train_residuals = train_target - arima_model.fittedvalues
        
        X_train_raw, y_train_raw, n_feat = self._create_3d_sequences(train_residuals, train_exog, train_target)
        
        # Escalamiento y Búsqueda LSTM
        if use_bayesian:
            try:
                import optuna
                from sklearn.metrics import accuracy_score
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                print(f"  🔍 [FASE 2] Optimizando LSTM Clasificador (Optuna Bayesiana TPE - Purged CV)...")
                tf.keras.backend.clear_session()

                scaler = StandardScaler()
                s_tr, t_steps, _ = X_train_raw.shape
                X_train_scaled = np.clip(
                    scaler.fit_transform(X_train_raw.reshape(-1, n_feat)), -10, 10
                ).reshape(s_tr, t_steps, n_feat)

                folds = self._get_purged_embargoed_folds(s_tr)

                def objective(trial):
                    units = trial.suggest_categorical('units', [32, 64, 128])
                    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)

                    fold_accs = []
                    for fold_idx, (train_idx, val_idx) in enumerate(folds):
                        if len(train_idx) == 0: continue
                        model_cv = self._build_lstm_classifier((t_steps, n_feat), units=units, dropout=dropout)
                        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
                        model_cv.fit(
                            X_train_scaled[train_idx], y_train_raw[train_idx],
                            epochs=15, batch_size=32,
                            validation_data=(X_train_scaled[val_idx], y_train_raw[val_idx]),
                            callbacks=[es], verbose=0, shuffle=True
                        )
                        preds = (model_cv.predict(X_train_scaled[val_idx], verbose=0) > 0.5).astype(int)
                        acc = accuracy_score(y_train_raw[val_idx], preds)
                        fold_accs.append(acc)

                        trial.report(acc, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    return float(np.mean(fold_accs)) if fold_accs else 0.0

                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

                best_lstm_params = study.best_params
                best_score = study.best_value
                print(f"  ✅ Ganador Bayesiano (Optuna ARIMA-LSTM): {best_lstm_params} (Acc Interno: {best_score:.2%})")
                return {'arima_order': best_arima_order, 'lstm_params': best_lstm_params}
            except Exception as e:
                print(f"  ⚠️ Fallback a ParameterSampler por error en Optuna: {e}")

        print(f"  🔍 [FASE 2] Optimizando LSTM Clasificador ({self.n_splits}-Fold Purged)...")
        tf.keras.backend.clear_session()
        
        scaler = StandardScaler()
        s_tr, t_steps, _ = X_train_raw.shape
        X_train_scaled = np.clip(
            scaler.fit_transform(X_train_raw.reshape(-1, n_feat)), -10, 10
        ).reshape(s_tr, t_steps, n_feat)

        folds = self._get_purged_embargoed_folds(s_tr)
        best_acc = -1
        
        from sklearn.model_selection import ParameterSampler
        param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))
        best_lstm_params = param_list[0]

        for params in param_list:
            fold_accs = []
            for train_idx, val_idx in folds:
                if len(train_idx) == 0: continue 
                
                model_cv = self._build_lstm_classifier((t_steps, n_feat), units=params.get('units', 64), dropout=params.get('dropout', 0.2))
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
                
                model_cv.fit(X_train_scaled[train_idx], y_train_raw[train_idx], epochs=15, batch_size=32, 
                             validation_data=(X_train_scaled[val_idx], y_train_raw[val_idx]), 
                             callbacks=[es], verbose=0, shuffle=True)
                
                preds = (model_cv.predict(X_train_scaled[val_idx], verbose=0) > 0.5).astype(int)
                fold_accs.append(accuracy_score(y_train_raw[val_idx], preds))
                
            avg_acc = np.mean(fold_accs) if fold_accs else 0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lstm_params = params

        print(f"  ✅ Ganador LSTM: {best_lstm_params} (Acc Interno: {best_acc:.2%})")
        return {'arima_order': best_arima_order, 'lstm_params': best_lstm_params}

    def walk_forward_predict(self, target: pd.Series, exog: pd.DataFrame, train_size: int, best_params: dict) -> tuple:

        """
        Ejecuta el Walk-Forward manteniendo vivos y actualizando DOS modelos 
        (ARIMA y LSTM) de forma paralela y estep-by-step.
        """
        print(f"  🚀 [FASE 3] Entrenando Maestros y ejecutando Walk-Forward...")
        tf.keras.backend.clear_session()
        
        train_target = target.iloc[:train_size]
        test_target = target.iloc[train_size:]
        train_exog = exog.iloc[:train_size] if exog is not None and not exog.empty else pd.DataFrame()
        test_exog = exog.iloc[train_size:] if exog is not None and not exog.empty else pd.DataFrame()

        # 1. Entrenar ARIMA Maestro Inicial
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMA(train_target, order=best_params['arima_order']).fit()
        train_residuals = train_target - arima_model.fittedvalues

        # 2. Preparar Dataset Inicial LSTM
        X_train_raw, y_train_raw, n_feat = self._create_3d_sequences(train_residuals, train_exog, train_target)
        
        scaler = StandardScaler()
        s_tr, t_steps, _ = X_train_raw.shape
        X_train_scaled = np.clip(scaler.fit_transform(X_train_raw.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)

        # 3. Entrenar LSTM Maestro Inicial
        lstm_params = best_params['lstm_params']
        master_lstm = self._build_lstm_classifier((t_steps, n_feat), units=lstm_params['units'], dropout=lstm_params['dropout'])
        master_lstm.fit(X_train_scaled, y_train_raw, epochs=20, batch_size=32, verbose=0, shuffle=True)

        # Variables de estado vivas
        pred_probs = []
        history_target = list(train_target.values)
        history_residuals = list(train_residuals.values)
        history_exog = train_exog.values.tolist() if not train_exog.empty else []
        
        X_train_list = list(X_train_raw)
        y_train_list = list(y_train_raw)
        model_arima_curr = arima_model

        # 4. Iteración Walk-Forward
        for i in range(len(test_target)):
            # Construir ventana actual
            last_resids = np.array(history_residuals[-self.look_back:]).reshape(-1, 1)
            if not test_exog.empty:
                last_exogs = np.array(history_exog[-self.look_back:])
                current_window = np.hstack([last_resids, last_exogs])
            else:
                current_window = last_resids
                
            # Escalar e inferir
            curr_win_scaled = np.clip(scaler.transform(current_window.reshape(-1, n_feat)), -10, 10).reshape(1, self.look_back, n_feat)
            prob_up = master_lstm.predict(curr_win_scaled, verbose=0)[0][0]
            pred_probs.append(prob_up)
            
            # Revelar la realidad y actualizar historia ARIMA
            actual_val = test_target.values[i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_arima = model_arima_curr.forecast(steps=1).iloc[0]
                    actual_resid = actual_val - pred_arima
                    model_arima_curr = model_arima_curr.append([actual_val], refit=False)
            except Exception:
                actual_resid = 0 # Fallback si el ARIMA diverge
                
            # Actualizar memorias
            X_train_list.append(current_window)
            y_train_list.append(1 if actual_val > history_target[-1] else 0)
            history_target.append(actual_val)
            history_residuals.append(actual_resid)
            if not test_exog.empty:
                history_exog.append(test_exog.values[i])
                
            # Re-entrenamiento periódico del LSTM
            if (i + 1) % self.retrain_step == 0:
                if (i+1) % (self.retrain_step*2) == 0:
                    print(f"    > Re-calibrando pesos Híbridos en paso {i+1}/{len(test_target)}...")
                
                curr_X_arr = np.array(X_train_list)
                curr_scaler = StandardScaler()
                curr_X_scaled = np.clip(
                    curr_scaler.fit_transform(curr_X_arr.reshape(-1, n_feat)), -10, 10
                ).reshape(-1, self.look_back, n_feat)
                
                master_lstm.fit(curr_X_scaled, np.array(y_train_list), epochs=2, batch_size=32, verbose=0, shuffle=True)
                scaler = curr_scaler 

        self.model_arima_curr = model_arima_curr
        self.master_lstm = master_lstm
        self.scaler = scaler # guardamos el último scaler usado
        self.best_params = best_params
        return np.array(pred_probs), None

    def save(self, filepath: str) -> None:
        """Saves the final trained models (ARIMA and LSTM), scaler and params."""
        if not hasattr(self, 'master_lstm') or not hasattr(self, 'model_arima_curr'):
            raise ValueError("No model trained yet. Run walk_forward_predict first.")
            
        keras_path = filepath.replace(".pkl", ".keras")
        self.master_lstm.save(keras_path)
        
        state = {
            'scaler': self.scaler,
            'look_back': self.look_back,
            'model_arima_curr': self.model_arima_curr,
            'best_params': getattr(self, 'best_params', None)
        }
        joblib.dump(state, filepath)
        print(f"  💾 ARIMA-LSTM model saved to {filepath} and {keras_path}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved model."""
        state = joblib.load(filepath)
        keras_path = filepath.replace(".pkl", ".keras")
        
        instance = cls(look_back=state['look_back'])
        instance.scaler = state['scaler']
        instance.model_arima_curr = state['model_arima_curr']
        instance.best_params = state.get('best_params', None)
        instance.master_lstm = tf.keras.models.load_model(keras_path)
        return instance
        
    def fast_retrain(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'close_FFD'):
        """Reajuste ligero de pesos neuronales sobre los residuos del ARIMA."""
        if getattr(self, 'master_lstm', None) is None:
            print("  ⚠️ No master_lstm found. Cannot fast retrain.")
            return
            
        df_valid = df.dropna(subset=[target_col])
        if len(df_valid) == 0:
            return
            
        train_target = df_valid[target_col].iloc[-1000:]
        train_exog = df_valid[feature_cols].iloc[-1000:] if feature_cols else pd.DataFrame()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima_order = self.best_params['arima_order'] if hasattr(self, 'best_params') and self.best_params else (1,0,1)
                temp_arima = ARIMA(train_target, order=arima_order).fit()
                train_residuals = train_target - temp_arima.fittedvalues
        except Exception:
            train_residuals = train_target
            
        X_train_raw, y_train_raw, n_feat = self._create_3d_sequences(train_residuals, train_exog, train_target)
        if len(X_train_raw) == 0:
            return
            
        s_tr, t_steps, _ = X_train_raw.shape
        X_train_scaled = np.clip(self.scaler.transform(X_train_raw.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)
        
        self.master_lstm.fit(X_train_scaled, y_train_raw, epochs=2, batch_size=32, verbose=0, shuffle=True)
        print("  ✅ Pesos de ARIMA-LSTM reentrenados exitosamente con datos recientes (2 epochs).")