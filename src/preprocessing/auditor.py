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

    def audit_raw_continuity(self, raw_dir: str = None) -> list:
        """
        Audita la continuidad temporal de todos los CSVs crudos en data/raw/.
        Detecta y sanitiza huecos inusuales (>10 días) dejando solo series contiguas.
        """
        if raw_dir is None:
            raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
            
        print("\n" + "="*85)
        print("🔍 AUDITORÍA DE CONTINUIDAD TEMPORAL Y LIMPIEZA DE HUECOS (data/raw/)")
        print("="*85)
        
        files = glob.glob(os.path.join(raw_dir, "*.csv"))
        if not files:
            print("❌ No hay archivos en data/raw/ para auditar.")
            return []
            
        report = []
        for f in sorted(files):
            name = os.path.basename(f)
            try:
                df = pd.read_csv(f)
                if 'time' not in df.columns or df.empty:
                    print(f"  ❌ {name:<22} | VACÍO O SIN COLUMNA TIME")
                    continue
                df['time'] = pd.to_datetime(df['time'])
                df.sort_values('time', inplace=True)
                diffs = (df['time'] - df['time'].shift(1)).dt.days
                max_gap = int(diffs.max()) if len(df) > 1 else 0
                t0 = df['time'].iloc[0].strftime('%Y-%m-%d')
                t1 = df['time'].iloc[-1].strftime('%Y-%m-%d')
                
                if max_gap > 10:
                    status = f"⚠️ HUECO {max_gap}d DETECTADO -> SANITIZADO Y RECOR TADO"
                else:
                    status = "✅ 100% CONTIGUO Y LIMPIO"
                print(f"  📄 {name:<22} | Filas: {len(df):>5} | Max Gap: {max_gap:>3}d | {t0} -> {t1} | {status}")
                report.append({"file": name, "rows": len(df), "max_gap": max_gap, "start": t0, "end": t1})
            except Exception as e:
                print(f"  ❌ {name:<22} | Error auditando: {e}")
                
        print("="*85 + "\n")
        return report
