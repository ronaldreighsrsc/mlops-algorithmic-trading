import pandas as pd
import yfinance as yf
import requests
import warnings
import urllib3
import io
import logging
from datetime import datetime

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ChileanMacroExtractor:
    """
    Extrae variables macroeconómicas relevantes para activos expuestos a la economía chilena.
    Reemplaza la necesidad de mantener archivos CSV manualmente, descargando toda la historia 
    disponible (ej. desde el año 2000) dinámicamente desde diversas APIs.
    """
    def __init__(self):
        self.yf_tickers = {
            'FXI': 'FXI',        # China Large-Cap ETF
            'USDCLP': 'CLP=X',   # Dólar frente al Peso Chileno
            'Copper': 'HG=F',    # Futuros del Cobre
            'Yield10Y': '^TNX'   # Bono Tesoro US 10 Años
        }
    
    def _fetch_yfinance(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        df_merged = pd.DataFrame()
        for col_name, ticker in self.yf_tickers.items():
            try:
                # yfinance usa strings o datetime para start/end
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                df = yf.download(ticker, start=start_str, end=end_str, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        serie = df['Close'].iloc[:, 0] if len(df['Close'].shape) > 1 else df['Close']
                    else:
                        serie = df['Close']
                    
                    df_merged[col_name] = serie
            except Exception as e:
                logging.error(f"Error descargando {col_name} ({ticker}) desde Yahoo Finance: {e}")
        
        if not df_merged.empty:
            df_merged.index = df_merged.index.tz_localize(None).normalize()
        return df_merged

    def _fetch_tpm(self) -> pd.DataFrame:
        """Extrae la historia completa de la Tasa de Política Monetaria desde mindicador.cl"""
        try:
            # Para la historia completa mindicador no sirve con endpoint /tpm (solo trae últimos días o el año pasado pasando el año en URL).
            # Para extraer de forma robusta la historia desde 2000 usando la API pública gratuita:
            # Haremos un truco: iteraremos los años desde el 2000 en la API de mindicador
            
            current_year = datetime.now().year
            all_years = []
            
            # Nota: mindicador permite años hacia atrás
            for year in range(2000, current_year + 1):
                res = requests.get(f"https://mindicador.cl/api/tpm/{year}", verify=False)
                if res.status_code == 200:
                    data = res.json()
                    if 'serie' in data and len(data['serie']) > 0:
                        df_year = pd.DataFrame(data['serie'])
                        all_years.append(df_year)
            
            if not all_years:
                return pd.DataFrame()
                
            df_tpm = pd.concat(all_years, ignore_index=True)
            df_tpm['fecha'] = pd.to_datetime(df_tpm['fecha']).dt.tz_localize(None).dt.normalize()
            df_tpm.rename(columns={'fecha': 'Date', 'valor': 'TPM'}, inplace=True)
            df_tpm.set_index('Date', inplace=True)
            df_tpm.sort_index(inplace=True)
            return df_tpm[['TPM']]
        except Exception as e:
            logging.error(f"Error extrayendo TPM: {e}")
            return pd.DataFrame()

    def _fetch_embi(self) -> pd.DataFrame:
        """Descarga el Excel oficial del BCRP (Banco Central) para obtener el EMBI Chile."""
        url = "https://bcrdgdcprod.blob.core.windows.net/documents/entorno-internacional/documents/Serie_Historica_Spread_del_EMBI.xlsx"
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            
            # Leer el archivo Excel en memoria
            df_embi = pd.read_excel(io.BytesIO(response.content), skiprows=1)
            df_embi = df_embi[['Fecha', 'Chile']].copy()
            df_embi.rename(columns={'Fecha': 'Date', 'Chile': 'EMBI'}, inplace=True)
            
            df_embi['Date'] = pd.to_datetime(df_embi['Date'], errors='coerce').dt.normalize()
            df_embi.dropna(subset=['Date'], inplace=True)
            df_embi.set_index('Date', inplace=True)
            df_embi.sort_index(inplace=True)
            
            # Limpieza
            df_embi['EMBI'] = pd.to_numeric(df_embi['EMBI'], errors='coerce')
            
            # Retraso operativo simulado de 3 días para EMBI para evitar Publication Lag en producción
            df_embi['EMBI'] = df_embi['EMBI'].shift(3)
            
            return df_embi[['EMBI']]
        except Exception as e:
            logging.error(f"Error descargando EMBI: {e}")
            return pd.DataFrame()

    def get_chilean_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logging.info("Extrayendo variables macro chilenas (FXI, USDCLP, Copper, Yield10Y, TPM, EMBI)...")
        
        # 1. Traer datos
        df_yf = self._fetch_yfinance(start_date, end_date)
        df_tpm = self._fetch_tpm()
        df_embi = self._fetch_embi()
        
        # 2. Base Index
        date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
        df_final = pd.DataFrame(index=date_range)
        df_final.index.name = 'time'
        
        # 3. Joins
        if not df_yf.empty:
            df_yf.index.name = 'time'
            df_final = df_final.join(df_yf, how='left')
        
        if not df_tpm.empty:
            df_tpm.index.name = 'time'
            # TPM es un valor constante hasta que cambia, usar ffill riguroso
            df_final = df_final.join(df_tpm, how='left')
            df_final['TPM'] = df_final['TPM'].ffill()
            
        if not df_embi.empty:
            df_embi.index.name = 'time'
            df_final = df_final.join(df_embi, how='left')
            
        # 4. Limpieza (ffill general para fines de semana o feriados desfasados)
        cols = df_final.columns
        if len(cols) > 0:
            df_final[cols] = df_final[cols].ffill()
            
            # Backward fill en caso de que empiece con NaN (debido al start_date)
            df_final[cols] = df_final[cols].bfill()
            
        return df_final
