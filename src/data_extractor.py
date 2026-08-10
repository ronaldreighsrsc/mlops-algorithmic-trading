import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import datetime as dt
from typing import Optional, List, Dict
import logging
from mt5_connector import MT5Connector
import time
import yfinance as yf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.chilean_macro import ChileanMacroExtractor

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExtractor:
    def __init__(self, connector: MT5Connector):
        self.connector = connector
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_dir = os.path.join(base_dir, "data", "raw")


    def _fetch_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logging.info("Descargando datos Macro (VIX, DXY, Yield10Y) usando yfinance...")
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
            tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
            
            macro_df = pd.DataFrame()
            if not vix.empty:
                # Si yfinance retorna MultiIndex en columnas
                if isinstance(vix.columns, pd.MultiIndex):
                    macro_df['VIX_close'] = vix['Close'].iloc[:, 0] if len(vix['Close'].shape) > 1 else vix['Close']
                else:
                    macro_df['VIX_close'] = vix['Close']
                    
            if not dxy.empty:
                if isinstance(dxy.columns, pd.MultiIndex):
                    macro_df['DXY_close'] = dxy['Close'].iloc[:, 0] if len(dxy['Close'].shape) > 1 else dxy['Close']
                else:
                    macro_df['DXY_close'] = dxy['Close']
                    
            if not tnx.empty:
                if isinstance(tnx.columns, pd.MultiIndex):
                    macro_df['Yield10Y'] = tnx['Close'].iloc[:, 0] if len(tnx['Close'].shape) > 1 else tnx['Close']
                else:
                    macro_df['Yield10Y'] = tnx['Close']
                    
            if not macro_df.empty:
                macro_df.index = macro_df.index.tz_localize(None).normalize()
            return macro_df
        except Exception as e:
            logging.error(f"Error descargando datos Macro: {e}")
            return pd.DataFrame()

    def get_historical_data_chunked(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Descarga datos dividiéndolos en pedazos de 1 año (Chunks) para burlar el timeout de los brokers.
        """
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Fallo al seleccionar el símbolo {symbol}.")
            return pd.DataFrame()

        logging.info(f"Descargando {symbol} en 'Chunks' anuales desde {start_date.year}...")
        all_rates = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = current_start + dt.timedelta(days=365)
            if current_end > end_date:
                current_end = end_date
                
            rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
            
            if rates is not None and len(rates) > 0:
                all_rates.append(pd.DataFrame(rates))
            else:
                logging.warning(f"  > Sin datos para {symbol} en el chunk {current_start.year}.")
                
            # Avanzar el cursor un día para no duplicar la fecha de corte
            current_start = current_end + dt.timedelta(days=1)
            time.sleep(0.1) # Pequeña pausa para no martillar el servidor
            
        if not all_rates:
            logging.error(f"Descarga final fallida para {symbol}.")
            return pd.DataFrame()

        df = pd.concat(all_rates, ignore_index=True)
        df.drop_duplicates(subset=['time'], inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Para Daily (D1) normalizamos a las 00:00:00. Para H4/H1 preservamos la hora exacta.
        if timeframe == mt5.TIMEFRAME_D1:
            df.index = df.index.normalize()
            
        df = df[['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']]
        
        # Merge con datos macro (VIX, DXY, Yield10Y)
        macro_df = self._fetch_macro_data(start_date, end_date)
        if not macro_df.empty:
            if timeframe == mt5.TIMEFRAME_D1:
                df = df.join(macro_df, how='left')
            else:
                # Para H4/H1, unimos por fecha y propagamos forward-fill por barra intradía
                df['date_key'] = df.index.date
                macro_df_temp = macro_df.copy()
                macro_df_temp['date_key'] = macro_df_temp.index.date
                macro_df_temp = macro_df_temp.drop_duplicates(subset=['date_key'])
                df = df.reset_index().merge(macro_df_temp, on='date_key', how='left').set_index('time')
                df.drop(columns=['date_key'], inplace=True, errors='ignore')

            # Forward fill para feriados o barras intradía
            for macro_col in ['VIX_close', 'DXY_close', 'Yield10Y']:
                if macro_col in df.columns:
                    df[macro_col] = df[macro_col].ffill().bfill()

            df['VIX_close'] = df['VIX_close'].bfill()
        if 'DXY_close' in df.columns:
            df['DXY_close'] = df['DXY_close'].bfill()
        if 'Yield10Y' in df.columns:
            df['Yield10Y'] = df['Yield10Y'].ffill().bfill()
            
        logging.info(f"Éxito: Se extrajeron {len(df)} registros totales para {symbol}.")
        return df

    def get_historical_data_yfinance(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logging.info(f"Descargando datos históricos de {symbol} vía yfinance...")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                return pd.DataFrame()
            
            # Aplanar columnas MultiIndex de yfinance >= 0.2
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            
            df.rename(columns={'volume': 'real_volume', 'adj close': 'adj_close'}, inplace=True)
            df['tick_volume'] = df['real_volume']
            df['spread'] = 0.0 # No spread info
            
            df.index = df.index.tz_localize(None).normalize()
            df.index.name = 'time'
            
            # Merge with macro data (VIX & DXY & Yield10Y)
            macro_df = self._fetch_macro_data(start_date, end_date)
            if not macro_df.empty:
                df = df.join(macro_df, how='left')
                if 'VIX_close' in df.columns: df['VIX_close'] = df['VIX_close'].ffill().bfill()
                if 'DXY_close' in df.columns: df['DXY_close'] = df['DXY_close'].ffill().bfill()
                if 'Yield10Y' in df.columns: df['Yield10Y'] = df['Yield10Y'].ffill().bfill()
                
            # MERGE WITH CHILEAN MACRO DATA FOR ECH
            if symbol == "ECH":
                macro_chile = ChileanMacroExtractor()
                df_chile = macro_chile.get_chilean_macro_data(start_date, end_date)
                if not df_chile.empty:
                    cols_to_use = [c for c in df_chile.columns if c not in df.columns]
                    if cols_to_use:
                        df = df.join(df_chile[cols_to_use], how='left')
                        # Forward fill para feriados
                        df[cols_to_use] = df[cols_to_use].ffill().bfill()

                
            logging.info(f"Éxito: Se extrajeron {len(df)} registros totales para {symbol} vía yfinance.")
            df.reset_index(inplace=True)
            # Asegurar retorno dinámico de columnas (para que soporte las nuevas que se añadieron)
            return df
        except Exception as e:
            logging.error(f"Error descargando {symbol} por yfinance: {e}")
            return pd.DataFrame()

    def get_existing_last_date(self, filename: str) -> Optional[datetime]:
        """
        Si el archivo CSV ya existe en data/raw/, retorna la última fecha registrada
        para hacer una actualización incremental rápida en lugar de re-descargar años de datos.
        """
        file_path = os.path.join(self.raw_dir, filename)
        if os.path.exists(file_path):
            try:
                df_old = pd.read_csv(file_path)
                if 'time' in df_old.columns and not df_old.empty:
                    df_old['time'] = pd.to_datetime(df_old['time'])
                    last_dt = df_old['time'].max()
                    # Si el formato contiene hora/minuto, restar 1 día para seguridad
                    return last_dt.to_pydatetime()
            except Exception as e:
                logging.warning(f"No se pudo leer la última fecha de {filename}: {e}")
        return None

    @staticmethod
    def sanitize_continuous_segment(df: pd.DataFrame, max_gap_days: int = 30) -> pd.DataFrame:
        """
        Detecta saltos temporales (huecos en la historia del broker) superiores a max_gap_days
        y conserva únicamente la ventana continua más reciente para proteger la integridad del FFD y ML.
        """
        if df.empty or 'time' not in df.columns:
            return df
            
        df_sorted = df.copy()
        df_sorted['time'] = pd.to_datetime(df_sorted['time'])
        df_sorted.sort_values('time', inplace=True)
        
        diffs = (df_sorted['time'] - df_sorted['time'].shift(1)).dt.days
        gap_indices = df_sorted.index[diffs > max_gap_days].tolist()
        
        if gap_indices:
            last_gap_idx = gap_indices[-1]
            df_clean = df_sorted.loc[last_gap_idx:].copy()
            t_start = df_clean['time'].iloc[0].strftime('%Y-%m-%d')
            t_end = df_clean['time'].iloc[-1].strftime('%Y-%m-%d')
            logging.info(f"🧹 [SANITIZADOR DE HUECOS] Se detectó un hueco > {max_gap_days} días. Dataset recortado a la serie contigua más reciente: {len(df_clean)} filas ({t_start} a {t_end}).")
            return df_clean
            
        return df_sorted

    def save_to_csv(self, df_new: pd.DataFrame, filename: str):
        if df_new.empty:
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        file_path = os.path.join(self.raw_dir, filename)
        
        if os.path.exists(file_path):
            try:
                df_old = pd.read_csv(file_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                if 'time' in df_combined.columns:
                    df_combined['time'] = pd.to_datetime(df_combined['time'])
                    df_combined.sort_values('time', inplace=True)
                    df_combined.drop_duplicates(subset=['time'], keep='last', inplace=True)
                
                df_combined = self.sanitize_continuous_segment(df_combined)
                df_combined.to_csv(file_path, index=False)
                logging.info(f"⚡ [UPDATE INCREMENTAL] Actualizado {filename} (Total: {len(df_combined)} registros contiguos).")
                return
            except Exception as e:
                logging.warning(f"Error fusionando datos incrementales para {filename}, se sobrescribirá: {e}")
                
        df_new = self.sanitize_continuous_segment(df_new)
        df_new.to_csv(file_path, index=False)
        logging.info(f"Datos guardados exitosamente en: {file_path}")

if __name__ == "__main__":
    from preprocessing.asset_screener import AssetScreener
    
    # 🔍 PASO 0: Screening Cuantitativo de Universo Dinámico (Descubrimiento MT5 + Hurst R/S + Descorrelación)
    conn_screen = MT5Connector()
    if conn_screen.connect():
        screener = AssetScreener(conn_screen)
        screen_res = screener.screen_universe(candidates=None, min_hurst=0.55, max_corr=0.80)
        target_extractions = screen_res.get("items_finales", [])
        conn_screen.shutdown()
    else:
        target_extractions = [
            {"nombre": "EURUSD", "ticker": "EURUSD", "timeframe": mt5.TIMEFRAME_D1, "filename": "EURUSD_daily.csv"},
            {"nombre": "EURUSD_H4", "ticker": "EURUSD", "timeframe": mt5.TIMEFRAME_H4, "filename": "EURUSD_H4_daily.csv"},
            {"nombre": "SP500", "ticker": "SP500", "timeframe": mt5.TIMEFRAME_D1, "filename": "SP500_daily.csv"},
            {"nombre": "SP500_H4", "ticker": "SP500", "timeframe": mt5.TIMEFRAME_H4, "filename": "SP500_H4_daily.csv"},
            {"nombre": "Oro", "ticker": "XAUUSD", "timeframe": mt5.TIMEFRAME_D1, "filename": "Oro_daily.csv"},
            {"nombre": "Oro_H4", "ticker": "XAUUSD", "timeframe": mt5.TIMEFRAME_H4, "filename": "Oro_H4_daily.csv"},
        ]
    
    end_dt = datetime.now()
    
    # TRUCO DEL USUARIO: Reconectar (Abrir y Cerrar) por cada activo/timeframe
    for item in target_extractions:
        nombre = item["nombre"]
        ticker = item["ticker"]
        tf = item["timeframe"]
        filename = item["filename"]
        
        conn = MT5Connector()
        if conn.connect():
            extractor = DataExtractor(conn)
            
            # ⚡ DETECCIÓN DE ACTUALIZACIÓN INCREMENTAL
            last_date = extractor.get_existing_last_date(filename)
            if last_date is not None:
                # Si el archivo ya existe, descargamos solo desde el último día registrado
                start_dt = max(datetime(2000, 1, 1), last_date - pd.Timedelta(days=1))
                logging.info(f"\n--- [INCREMENTAL] Actualizando {nombre} ({ticker}) desde {start_dt.strftime('%Y-%m-%d')} ---")
            else:
                # Descarga limpia desde el 2000
                start_dt = datetime(2000, 1, 1)
                logging.info(f"\n--- [DESCARGA COMPLETA] Extracción desde 2000 para {nombre} ({ticker}) ---")
                
            df_activo = extractor.get_historical_data_chunked(
                symbol=ticker, 
                timeframe=tf,
                start_date=start_dt, 
                end_date=end_dt
            )
            
            if not df_activo.empty:
                df_activo.reset_index(inplace=True)
                extractor.save_to_csv(df_activo, filename)
                
            conn.shutdown()
            
        time.sleep(0.5)

