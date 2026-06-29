import pandas as pd
import os
import glob
import logging

class DataAuditor:
    """
    Clase encargada de auditar la integridad matemática y estructural 
    de los datasets procesados antes de pasarlos a los modelos de Machine Learning.
    Cumple con el Principio de Responsabilidad Única (SRP) de SOLID.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def audit_all(self):
        print("\n" + "="*60)
        print("🔍 INICIANDO AUDITORÍA AUTOMÁTICA DE DATOS PROCESADOS")
        print("="*60)
        
        if not os.path.exists(self.data_dir):
            print("❌ Carpeta de datos procesados no encontrada.")
            return
            
        csv_files = glob.glob(os.path.join(self.data_dir, "*_processed.csv"))
        if not csv_files:
            print("❌ No hay archivos para auditar.")
            return
            
        for path in csv_files:
            self._audit_file(path)

        print("\n✅ Auditoría de Integridad Finalizada.")

    def _audit_file(self, file_path: str):
        activo = os.path.basename(file_path).replace("_processed.csv", "")
        df = pd.read_csv(file_path)
        
        print(f'\n{"-"*60}')
        print(f'📊 {activo} | Filas: {len(df)} | Columnas: {len(df.columns)}')
        print(f'{"-"*60}')
        
        self._check_nulls(df)
        self._check_zero_columns(df)
        self._check_date_range(df)

    def _check_nulls(self, df: pd.DataFrame):
        nulls = df.isnull().sum()
        cols_with_nulls = nulls[nulls > 0]
        if len(cols_with_nulls) > 0:
            print(f'⚠️  ALERTA: Columnas con NaN (Peligro para ML):')
            for col, n in cols_with_nulls.items():
                pct = n / len(df) * 100
                print(f'   - {col}: {n} NaN ({pct:.1f}%)')
        else:
            print(f'✅ Datos limpios: 0 Valores Nulos')

    def _check_zero_columns(self, df: pd.DataFrame):
        zero_cols = [c for c in df.select_dtypes(include='number').columns if (df[c] == 0).all()]
        if zero_cols:
            print(f'⚠️  Aviso: Columnas con 100% Ceros: {zero_cols} (Revisar si es normal, ej: spread o volumen)')

    def _check_date_range(self, df: pd.DataFrame):
        if 'time' in df.columns:
            print(f'📅 Período: {df["time"].iloc[0]} → {df["time"].iloc[-1]}')
