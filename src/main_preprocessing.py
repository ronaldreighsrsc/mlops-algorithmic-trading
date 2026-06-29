import os
import glob
import pandas as pd
import warnings
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from preprocessing.technical_features import TechnicalFeatureEngineer
from preprocessing.volatility import VolatilityModeler
from preprocessing.stationarity import FractionalDifferencer

warnings.filterwarnings("ignore")

def run_preprocessing_pipeline():
    print("🚀 Iniciando Pipeline de Preprocesamiento de Datos (Fase 2)...")
    
    # Buscar todos los archivos CSV descargados en la Fase 1
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    # Asegurarnos de que la carpeta de destino exista (usando ruta absoluta)
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    if not csv_files:
        print("❌ No se encontraron archivos en data/raw/. Por favor ejecuta data_extractor.py primero.")
        return

    # Iterar sobre cada activo extraído
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        ticker = filename.replace("_daily.csv", "")
        
        print(f"\n{'='*50}")
        print(f"📈 PROCESANDO ACTIVO: {ticker}")
        print(f"{'='*50}")
        
        # 1. Cargar Datos
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        # Convertir a datetime y ordenar por si acaso
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
        df.set_index('time', inplace=True)
        
        # Eliminar si hay NaNs en los precios (fines de semana, feriados raros)
        df.dropna(subset=['close'], inplace=True)

        print(f"\n--- PASO 1: Ingeniería de Features Técnicos ---")
        # El precio objetivo en MT5 siempre es 'close'
        engineer = TechnicalFeatureEngineer(target_price_col='close')
        df = engineer.add_indicators(df)

        print(f"\n--- PASO 2: Modelado de Volatilidad (EGARCH) ---")
        # Para swing trading diario, 500 días (aprox 2 años) es una buena ventana
        vol_modeler = VolatilityModeler(window_size=500, target_col='close')
        df = vol_modeler.compute_egarch(df)

        print(f"\n--- PASO 3: Estacionariedad y Memoria (FFD) ---")
        differencer = FractionalDifferencer(threshold=1e-4)
        df_final = differencer.apply_ffd(df)

        print(f"\n--- PASO 4: Triple Barrera (Meta-Labeling Y) ---")
        from preprocessing.meta_labeling import TripleBarrierLabeler
        
        # Parámetros dinámicos por activo
        if ticker == "Oro":
            k_up_val, k_down_val = 2.5, 1.5
        elif ticker == "SP500":
            k_up_val, k_down_val = 2.0, 1.5
        else: # EURUSD y por defecto
            k_up_val, k_down_val = 2.0, 1.5
            
        print(f"Configurando barreras: k_up={k_up_val}, k_down={k_down_val}")
        labeler = TripleBarrierLabeler(k_up=k_up_val, k_down=k_down_val, max_hold=10)
        df_final = labeler.apply_labels(df_final)

        print(f"\n--- PASO 5: Guardando Resultados ---")
        output_filename = f"{ticker}_processed.csv"
        output_path = os.path.join(base_dir, "data", "processed", output_filename)
        
        # En MT5 tenemos 'time' como índice, lo guardaremos como columna de fecha
        df_final.reset_index(inplace=True)
        df_final.to_csv(output_path, index=False)
        
        print(f"✅ ¡Pipeline completado para {ticker}!")
        print(f"📁 Datos limpios y estacionarios guardados en: {output_path}")

    print("\n🎉 TODOS LOS ACTIVOS PROCESADOS CON ÉXITO.")

def audit_processed_data():
    print("\n" + "="*50)
    print("🔍 INICIANDO AUDITORÍA DE DATOS PROCESADOS")
    print("="*50)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    assets = ['EURUSD', 'SP500', 'Oro', 'ECH']
    
    for activo in assets:
        path = os.path.join(data_dir, f'{activo}_processed.csv')
        if not os.path.exists(path):
            print(f'\n❌ {activo}: ARCHIVO NO ENCONTRADO')
            continue
            
        df = pd.read_csv(path)
        print(f'\n{"="*70}')
        print(f'📊 {activo} | Filas: {len(df)} | Columnas: {len(df.columns)}')
        print(f'{"="*70}')
        
        # Check nulls
        nulls = df.isnull().sum()
        cols_with_nulls = nulls[nulls > 0]
        if len(cols_with_nulls) > 0:
            print(f'⚠️  Columnas con NaN:')
            for col, n in cols_with_nulls.items():
                pct = n/len(df)*100
                print(f'   {col}: {n} NaN ({pct:.1f}%)')
        else:
            print(f'✅ Sin NaN en ninguna columna')
        
        # Check columns with all zeros
        zero_cols = [c for c in df.select_dtypes(include='number').columns if (df[c] == 0).all()]
        if zero_cols:
            print(f'❌ Columnas 100% CEROS: {zero_cols}')
        
        # Check columns with constant values  
        const_cols = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() <= 1]
        if const_cols:
            print(f'⚠️  Columnas CONSTANTES (1 valor): {const_cols}')
        
        # Check date range
        if 'time' in df.columns:
            print(f'📅 Rango: {df["time"].iloc[0]} → {df["time"].iloc[-1]}')
        
        # Print all columns with basic stats
        print(f'\n   {"Col":<25} {"Min":>12} {"Max":>12} {"Mean":>12} {"Std":>12} {"NaN":>6}')
        print(f'   {"-"*25} {"-"*12} {"-"*12} {"-"*12} {"-"*12} {"-"*6}')
        for col in df.select_dtypes(include='number').columns:
            mn = df[col].min()
            mx = df[col].max()
            me = df[col].mean()
            st = df[col].std()
            na = df[col].isna().sum()
            print(f'   {col:<25} {mn:>12.4f} {mx:>12.4f} {me:>12.4f} {st:>12.4f} {na:>6}')

if __name__ == "__main__":
    run_preprocessing_pipeline()
    audit_processed_data()

