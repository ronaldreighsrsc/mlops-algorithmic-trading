import os
import sys
import pandas as pd
import numpy as np
import warnings

# Forzar UTF-8 y flush inmediato en la consola de Windows
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'
from sklearn.metrics import accuracy_score
import tensorflow as tf

from models.arimax import ARIMAXTrainer
from models.random_forest import RandomForestTrainer
from models.xgb_model import XGBoostTrainer
from models.lstm_model import LSTMTrainer
from models.bilstm_model import BiLSTMTrainer
from models.arima_lstm import HybridARIMALSTMTrainer
from models.lstm_rf import HybridLSTMRFTrainer

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==============================================================================
# ⚠️ MODO RECUPERACIÓN (Crash RAM): Corriendo de a 1 activo.
ACTIVOS_A_CORRER = ["Oro"] # ["ECH", "EURUSD", "SP500", "Oro"]
MODELOS_A_CORRER = [
    #'ARIMAX', 
    'RANDOM_FOREST',
    'XGBOOST', 
    'LSTM',
    'BILSTM',
    'ARIMA_LSTM',
    'LSTM_RF' # Solo falta este para terminar EURUSD
]

# Grillas de Hiperparámetros (Distribuciones para Randomized Search)
RF_GRID = {'n_estimators': [100, 250, 500, 1000], 'max_depth': [5, 10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'max_features': ['sqrt', 'log2']}
XGB_GRID = {'n_estimators': [200, 500, 1000], 'max_depth': [3, 5, 7, 9], 'learning_rate': [0.005, 0.01, 0.05, 0.1], 'subsample': [0.6, 0.8, 1.0]}
NN_GRID = {'units': [32, 64, 128, 256], 'dropout': [0.1, 0.2, 0.3]}
HIBRIDO_RF_GRID = {'units': [32, 64, 128], 'dropout': [0.1, 0.2, 0.3]}

# Bancos de Variables Dinámicos por Activo
def get_bancos_por_activo(activo: str):
    precio_puro = ['open_FFD', 'high_FFD', 'low_FFD']
    tecnicos = ['MACD_Hist', 'RSI', 'ATR', 'EGARCH_Vol']
    
    if activo in ["ECH", "IPSA"]:
        macros = ['TPM', 'EMBI', 'Copper_FFD', 'Yield10Y_FFD', 'USDCLP_FFD']
        globales = ['SP500_FFD', 'VIX_close', 'FXI_FFD']
    elif activo in ["EURUSD", "SP500"]:
        macros = ['Yield10Y_FFD']
        globales = ['VIX_close', 'DXY_close_FFD']
    elif activo == "Oro":
        macros = ['Yield10Y_FFD', 'DXY_close_FFD'] 
        globales = ['SP500_FFD', 'VIX_close']
    else:
        macros = []
        globales = []

    bancos = {
        "Precio_Puro": precio_puro,
        "Precio_Volumen": precio_puro + ['tick_volume'],
        "Tecnicos": tecnicos,
        "Macros": macros,
        "Globales": globales,
        "Hibrido_Precio_Tec_Vol": precio_puro + tecnicos + ['tick_volume'],
        #"Kitchen_Sink_Total": precio_puro + tecnicos + macros + globales + ['tick_volume']
    }
    # Filtrar bancos vacíos
    return {k: v for k, v in bancos.items() if v}

def train_for_asset(activo: str):
    print(f"🚀 Iniciando Torneo de Modelos para {activo}...")

    # 1. Carga de datos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", f"{activo}_processed.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ No se encontró {data_path}.")
        
    df_base = pd.read_csv(data_path)
    if 'time' in df_base.columns:
        df_base['time'] = pd.to_datetime(df_base['time'])
        df_base = df_base.sort_values('time').set_index('time')
    
    df_base.dropna(inplace=True)

    results_dir = os.path.join(base_dir, "results")
    models_out_dir = os.path.join(results_dir, "saved_models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_out_dir, exist_ok=True)
    
    resultados_globales = []

    # 2. Iteración de Modelos
    for nombre_modelo in MODELOS_A_CORRER:
        print(f"\n{'#'*70}\n🏆 EVALUANDO MODELO: {nombre_modelo}\n{'#'*70}")
        
        # Instanciar el Trainer correspondiente
        if nombre_modelo == 'ARIMAX':
            trainer = ARIMAXTrainer(p_values=[1, 2], q_values=[1], retrain_step=60)
        elif nombre_modelo == 'RANDOM_FOREST':
            trainer = RandomForestTrainer(look_back=60, retrain_step=60)
        elif nombre_modelo == 'XGBOOST':
            trainer = XGBoostTrainer(look_back=60, retrain_step=60)
        elif nombre_modelo == 'LSTM':
            trainer = LSTMTrainer(look_back=60, retrain_step=60)
        elif nombre_modelo == 'BILSTM':
            trainer = BiLSTMTrainer(look_back=60, retrain_step=60)
        elif nombre_modelo == 'ARIMA_LSTM':
            trainer = HybridARIMALSTMTrainer(look_back=60, retrain_step=60, p_values=[1,2], q_values=[1])
        elif nombre_modelo == 'LSTM_RF':
            trainer = HybridLSTMRFTrainer(look_back=60, retrain_step=60)
        else:
            continue

        # Target Check (Meta-Labeling vs Econometría)
        if nombre_modelo in ['ARIMAX', 'ARIMA_LSTM']:
            target_col = 'close_FFD'  # Los econométricos siguen usando el retorno continuo FFD
        else:
            target_col = 'Label'      # Los ML/DL usan la nueva etiqueta institucional Triple Barrera

        # 3. Iteración de Bancos
        bancos_activo = get_bancos_por_activo(activo)
        for nombre_banco, features in bancos_activo.items():
            print(f"\n{'-'*50}\n🧠 BANCO: {nombre_banco} | MODELO: {nombre_modelo} | TARGET: {target_col}\n{'-'*50}")
            
            valid_features = [c for c in features if c in df_base.columns]
            if not valid_features and nombre_modelo not in ['ARIMAX', 'ARIMA_LSTM']:
                print(f"  ⚠️ [SKIP] El banco '{nombre_banco}' no contiene variables válidas para el activo {activo} en el modelo {nombre_modelo}.")
                continue
            
            # --- RUTAS DE EJECUCIÓN SEGÚN INTERFAZ DEL MODELO ---
            if nombre_modelo in ['ARIMAX', 'ARIMA_LSTM']:
                if valid_features:
                    df_exog_lagged = df_base[valid_features].shift(1)
                    df_model = pd.concat([df_base[target_col], df_exog_lagged], axis=1).dropna()
                    exog = df_model[valid_features]
                else:
                    df_model = df_base[[target_col]].dropna()
                    exog = pd.DataFrame()
                    
                target = df_model[target_col]
                train_size = int(len(target) * 0.8)
                
                if nombre_modelo == 'ARIMAX':
                    best_params = trainer.find_best_order(target.iloc[:train_size], exog.iloc[:train_size])
                    # Generar In-Sample Probs (aproximado usando el modelo ajustado)
                    import statsmodels.api as sm
                    try:
                        temp_model = sm.tsa.ARIMA(target.iloc[:train_size], exog=exog.iloc[:train_size] if not exog.empty else None, order=best_params).fit()
                        train_probs = temp_model.predict(start=0, end=train_size-1).apply(lambda x: 1 if x > 0 else 0).values
                    except:
                        train_probs = np.zeros(train_size)
                    pred_probs = trainer.walk_forward_predict(target, exog, train_size, best_params)
                else: # ARIMA_LSTM
                    best_params = trainer.find_best_params(target.iloc[:train_size], exog.iloc[:train_size], NN_GRID, n_iter=10)
                    pred_probs, _ = trainer.walk_forward_predict(target, exog, train_size, best_params)
                    
                    # Generar In-Sample Probs reales para ARIMA-LSTM
                    try:
                        arima_fitted = trainer.model_arima_curr.fittedvalues
                        train_resids_full = target.iloc[:train_size] - arima_fitted.iloc[:train_size]
                        X_tr_raw, y_tr_raw, n_f = trainer._create_3d_sequences(
                            train_resids_full, 
                            exog.iloc[:train_size] if not exog.empty else None, 
                            target.iloc[:train_size]
                        )
                        s_tr, t_steps, _ = X_tr_raw.shape
                        X_tr_scaled = np.clip(trainer.scaler.transform(X_tr_raw.reshape(-1, n_f)), -10, 10).reshape(s_tr, t_steps, n_f)
                        in_sample_probs = trainer.master_lstm.predict(X_tr_scaled, verbose=0)[:, 0]
                        train_probs = np.concatenate([np.zeros(train_size - len(in_sample_probs)), in_sample_probs])
                    except Exception as e:
                        print(f"⚠️ Error al generar train_probs para ARIMA_LSTM: {e}")
                        train_probs = np.zeros(train_size)
                    
            else:
                if nombre_modelo == 'RANDOM_FOREST':
                    grid = RF_GRID
                elif nombre_modelo == 'XGBOOST':
                    grid = XGB_GRID
                elif nombre_modelo == 'LSTM_RF':
                    grid = HIBRIDO_RF_GRID
                else:
                    grid = NN_GRID
                    
                X, y = trainer.prepare_sequences(df_base, target_col, valid_features)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                best_params = trainer.find_best_params(X_train, y_train, grid, n_iter=10)
                
                # Ejecutar Walk-Forward Predict primero para entrenar y guardar los modelos en la instancia
                pred_probs, importances = trainer.walk_forward_predict(X_train, y_train, X_test, y_test, best_params)
                
                # Generar In-Sample Probs reales sin el hack dummy
                if nombre_modelo in ['RANDOM_FOREST', 'XGBOOST']:
                    X_train_scaled = np.clip(trainer.scaler.transform(X_train), -10, 10)
                    train_probs = trainer.master_model.predict_proba(X_train_scaled)[:, 1]
                elif nombre_modelo in ['LSTM', 'BILSTM']:
                    s_tr, t_steps, n_feat = X_train.shape
                    X_train_scaled = np.clip(trainer.scaler.transform(X_train.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)
                    train_probs = trainer.master_model.predict(X_train_scaled, verbose=0)[:, 0]
                elif nombre_modelo == 'LSTM_RF':
                    s_tr, t_steps, n_feat = X_train.shape
                    X_train_scaled = np.clip(trainer.scaler.transform(X_train.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)
                    X_train_features = trainer.feature_extractor.predict(X_train_scaled, verbose=0)
                    train_probs = trainer.rf_model.predict_proba(X_train_features)[:, 1]
                else:
                    train_probs = y_train
                
            # Guardar Modelo
            try:
                save_path = os.path.join(models_out_dir, f"{nombre_modelo.lower()}_{nombre_banco.lower()}_{activo}.pkl")
                trainer.save(save_path)
            except Exception as e:
                print(f"⚠️ No se pudo guardar el modelo: {e}")
            
            # Guardar Probabilidades OOS (Test)
            npy_path = os.path.join(results_dir, f"probs_{nombre_modelo.lower()}_{nombre_banco.lower()}_{activo}.npy")
            np.save(npy_path, np.array(pred_probs))
            
            # Guardar Probabilidades In-Sample (Train)
            if 'train_probs' in locals():
                npy_train_path = os.path.join(results_dir, f"train_probs_{nombre_modelo.lower()}_{nombre_banco.lower()}_{activo}.npy")
                np.save(npy_train_path, np.array(train_probs))
            
            resultados_globales.append({
                "Activo": activo,
                "Modelo": nombre_modelo,
                "Banco": nombre_banco,
                "Mejores_Params": str(best_params)
            })
            
            # 🧹 Limpiar Memoria RAM y Backend Keras (Garbage Collection Crítico)
            import gc
            tf.keras.backend.clear_session()
            gc.collect()

    # 4. Resumen Global
    if resultados_globales:
        df_new = pd.DataFrame(resultados_globales)
        csv_path = os.path.join(results_dir, 'tabla_entrenamiento_global.csv')
        
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_new, df_old]).drop_duplicates(subset=['Activo', 'Modelo', 'Banco'], keep='first')
        else:
            df_combined = df_new
            
        print("\n" + "="*60 + "\n🏆 TABLA DE ENTRENAMIENTO 🏆\n" + "="*60)
        print(df_combined.to_string(index=False))
        df_combined.to_csv(csv_path, index=False)
        print(f"\n✅ Pipeline Completado para {activo}.")

def run_training_pipeline():
    for activo in ACTIVOS_A_CORRER:
        train_for_asset(activo)
    print("\n🎉 TODOS LOS ACTIVOS HAN SIDO PROCESADOS EXITOSAMENTE.")

if __name__ == "__main__":
    run_training_pipeline()
