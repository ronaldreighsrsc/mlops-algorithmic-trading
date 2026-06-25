import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import datetime as dt
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

    def _fetch_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logging.info("Descargando datos Macro (VIX y DXY) usando yfinance...")
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
            
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
        df.index = df.index.normalize() # Normalizar para hacer match correcto con yfinance
        df = df[['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']]
        
        # Merge con datos macro
        macro_df = self._fetch_macro_data(start_date, end_date)
        if not macro_df.empty:
            df = df.join(macro_df, how='left')
            # Forward fill para dias donde forex opera pero mercados de bolsa/indices estan cerrados
            if 'VIX_close' in df.columns:
                df['VIX_close'] = df['VIX_close'].ffill()
            if 'DXY_close' in df.columns:
                df['DXY_close'] = df['DXY_close'].ffill()
        
        # Backward fill just in case the first rows are NaN
        if 'VIX_close' in df.columns:
            df['VIX_close'] = df['VIX_close'].bfill()
        if 'DXY_close' in df.columns:
            df['DXY_close'] = df['DXY_close'].bfill()
            
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
            
            # Merge with macro data (VIX & DXY)
            macro_df = self._fetch_macro_data(start_date, end_date)
            if not macro_df.empty:
                df = df.join(macro_df, how='left')
                if 'VIX_close' in df.columns: df['VIX_close'] = df['VIX_close'].ffill().bfill()
                if 'DXY_close' in df.columns: df['DXY_close'] = df['DXY_close'].ffill().bfill()
                
            # MERGE WITH CHILEAN MACRO DATA FOR ECH
            if symbol == "ECH":
                macro_chile = ChileanMacroExtractor()
                df_chile = macro_chile.get_chilean_macro_data(start_date, end_date)
                if not df_chile.empty:
                    df = df.join(df_chile, how='left')
                    # Forward fill para feriados
                    cols_chile = df_chile.columns
                    df[cols_chile] = df[cols_chile].ffill().bfill()
                
            logging.info(f"Éxito: Se extrajeron {len(df)} registros totales para {symbol} vía yfinance.")
            df.reset_index(inplace=True)
            # Asegurar retorno dinámico de columnas (para que soporte las nuevas que se añadieron)
            return df
        except Exception as e:
            logging.error(f"Error descargando {symbol} por yfinance: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        if df.empty:
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        file_path = os.path.join(self.raw_dir, filename)
        df.to_csv(file_path, index=False)
        logging.info(f"Datos guardados exitosamente en: {file_path}")

if __name__ == "__main__":
    activos = {
        "SP500": "SP500", 
        "EURUSD": "EURUSD", 
        "Oro": "XAUUSD",
        "ECH": "ECH"
    }
    
    # Intentamos desde el año 2000
    end_dt = datetime.now()
    start_dt = datetime(2000, 1, 1)
    
    # TRUCO DEL USUARIO: Reconectar (Abrir y Cerrar) por cada activo
    for nombre, ticker in activos.items():
        logging.info(f"\n--- Iniciando ciclo de extracción aislado para {ticker} ---")
        
        # Para ECH, saltar MT5 e ir directo a yfinance
        if ticker == "ECH":
            extractor = DataExtractor(None)
            df_activo = extractor.get_historical_data_yfinance(ticker, start_dt, end_dt)
            if not df_activo.empty:
                # ECH usa el índice como time, lo reseteamos para guardar igual que MT5
                df_activo.reset_index(inplace=True)
                extractor.save_to_csv(df_activo, f"{nombre}_daily.csv")
        else:
            conn = MT5Connector()
            if conn.connect():
                extractor = DataExtractor(conn)
                df_activo = extractor.get_historical_data_chunked(
                    symbol=ticker, 
                    timeframe=mt5.TIMEFRAME_D1,
                    start_date=start_dt, 
                    end_date=end_dt
                )
                
                if not df_activo.empty:
                    df_activo.reset_index(inplace=True)
                    extractor.save_to_csv(df_activo, f"{nombre}_daily.csv")
                    
                conn.shutdown() # Cerramos la conexión específica de este activo
            
        time.sleep(1) # Pausa técnica antes de abrir la siguiente conexión
