import MetaTrader5 as mt5
import logging
import os
from dotenv import load_dotenv

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MT5Connector:
    """
    Clase para gestionar la conexión con MetaTrader 5 (Darwinex).
    """
    def __init__(self, login: int = None, password: str = None, server: str = None):
        # Intentamos cargar desde .env si no se pasan parámetros
        load_dotenv()
        
        self.login = login if login else int(os.getenv("MT5_LOGIN", 0))
        self.password = password if password else os.getenv("MT5_PASSWORD")
        self.server = server if server else os.getenv("MT5_SERVER")
        self.connected = False

    def connect(self) -> bool:
        """
        Inicializa la conexión con la terminal de MT5 y, opcionalmente, 
        inicia sesión en la cuenta proporcionada de manera oculta.
        """
        logging.info("Inicializando conexión con MetaTrader 5...")
        
        # Inicializa la conexión. Si el terminal está cerrado, Python lo abrirá en segundo plano (silent mode).
        if not mt5.initialize():
            logging.error(f"Fallo al inicializar MT5. Error code: {mt5.last_error()}")
            return False
            
        logging.info("Terminal MT5 inicializada correctamente.")

        # Si hay credenciales configuradas, hacemos login automático
        if self.login and self.password and self.server:
            logging.info(f"Intentando login automático (oculto) en el servidor: {self.server}...")
            authorized = mt5.login(self.login, password=self.password, server=self.server)
            if not authorized:
                logging.error(f"Fallo al iniciar sesión. Verifica tus credenciales en el .env. Error code: {mt5.last_error()}")
                mt5.shutdown()
                return False
            logging.info("Login exitoso. Autenticado correctamente.")
        else:
            logging.warning("No se encontraron credenciales en el archivo .env. Solo se conectó al terminal local.")
            
        self.connected = True
        return True

    def get_account_info(self):
        """
        Obtiene y muestra la información de la cuenta conectada.
        """
        if not self.connected:
            logging.warning("No hay conexión activa con MT5.")
            return None
            
        account_info = mt5.account_info()
        if account_info is None:
            logging.error(f"No se pudo obtener información de la cuenta. Error code: {mt5.last_error()}")
            return None
            
        logging.info(f"=== Información de la Cuenta MT5 ===")
        logging.info(f"Login: {account_info.login}")
        logging.info(f"Broker: {account_info.company}")
        logging.info(f"Servidor: {account_info.server}")
        logging.info(f"Balance: {account_info.balance} {account_info.currency}")
        logging.info(f"Equidad: {account_info.equity}")
        return account_info

    def shutdown(self):
        """
        Cierra la conexión con MT5 de forma segura.
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Conexión con MT5 cerrada.")

if __name__ == "__main__":
    # Prueba de conexión rápida
    # NOTA: Debes tener el terminal de MT5 abierto en tu PC para que esto funcione.
    connector = MT5Connector()
    if connector.connect():
        connector.get_account_info()
        connector.shutdown()
