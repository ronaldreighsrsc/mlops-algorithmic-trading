import os
import requests
import logging
from dotenv import load_dotenv

class TelegramNotifier:
    """
    Clase encargada de enviar notificaciones al celular del usuario a través 
    de la API oficial de Telegram Bots.
    """
    def __init__(self, token: str = None, chat_id: str = None):
        load_dotenv()
        self.token = token if token else os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id if chat_id else os.getenv("TELEGRAM_CHAT_ID")
        
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            logging.warning("No se detectaron TELEGRAM_TOKEN o TELEGRAM_CHAT_ID. Las notificaciones móviles están deshabilitadas.")

    def send_message(self, message: str) -> bool:
        """
        Envía un mensaje de texto al chat configurado.
        """
        if not self.enabled:
            return False
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown" # Permite usar negritas (*texto*)
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logging.error(f"Fallo al enviar mensaje Telegram: {response.text}")
                return False
        except Exception as e:
            logging.error(f"Error de red enviando mensaje a Telegram: {e}")
            return False

    def alert_startup(self):
        msg = "🟢 *QuantBot Iniciado*\nEl sistema ha arrancado exitosamente en el servidor AWS.\nEsperando señales..."
        self.send_message(msg)
        
    def alert_daily_check(self, symbol: str, vol: float, has_signal: bool):
        signal_text = "Señal: ESPERAR ⏳" if not has_signal else "Señal: **DISPARADA** 🚀"
        msg = (
            f"📊 *Check Diario: {symbol}*\n"
            f"- Volatilidad (EGARCH): {vol:.2f}%\n"
            f"- {signal_text}\n\n"
            f"_Bot activo en AWS. Próxima revisión mañana ~5:15 PM (hora Chile)._"
        )
        self.send_message(msg)

    def alert_trade_execution(self, symbol: str, volume: float, price: float, tp: float, sl: float, is_long: bool = True, account_balance: float = 500.0, risk_pct: float = 0.01, timeframe: str = "D1"):
        # Calcular riesgo en dolares y pips para referencia
        sl_pips = abs(price - sl) * 10000
        tp_pips = abs(tp - price) * 10000
        # Lógica exclusiva para Quantfury (Manual)
        quantfury_balance = float(os.getenv("QUANTFURY_BALANCE", 2000.0))
        quantfury_max_power = float(os.getenv("QUANTFURY_MAX_POWER", 40000.0))
        
        riesgo_manual_usd = quantfury_balance * risk_pct
        
        if price != sl:
            porcentaje_movimiento_sl = abs(price - sl) / price
            quantfury_trading_power = riesgo_manual_usd / porcentaje_movimiento_sl
        else:
            quantfury_trading_power = 0.0
            
        warning_leverage = ""
        if quantfury_trading_power > quantfury_max_power:
            quantfury_trading_power = quantfury_max_power
            warning_leverage = f"\n⚠️ *MAX LEVERAGE ALCANZADO*: Limitado al tope de ${quantfury_max_power:,.2f}"
            
        direccion_str = "COMPRA (Long) 📈" if is_long else "VENTA (Short) 📉"
        
        msg = (
            f"🚀 *SEÑAL DE {direccion_str} — {symbol} [{timeframe}]*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Precio Entrada: `{price:.5f}`\n"
            f"🎯 Take Profit:    `{tp:.5f}` (+{tp_pips:.0f} pips)\n"
            f"🛡️ Stop Loss:      `{sl:.5f}` (-{sl_pips:.0f} pips)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Ejecución MT5 (Automática Demo/Real)*\n"
            f"💵 Balance Asumido: ${account_balance:.2f}\n"
            f"📦 Lotes inyectados en MT5: `{volume}` Lotes\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📱 *Ejecución Manual (Quantfury)*\n"
            f"💵 Balance Asumido: ${quantfury_balance:,.2f} USDT\n"
            f"⚠️ Riesgo Matemático a Perder: ${riesgo_manual_usd:,.2f} USDT\n"
            f"👉 _Poder de Trading:_ Escribe exactamente `$ {quantfury_trading_power:,.2f}` en Quantfury.{warning_leverage}"
        )
        self.send_message(msg)


    def alert_mlops_quarantine(self, symbol: str):
        msg = (
            f"🚨 *CUARENTENA MLOps ACTIVADA* 🚨\n"
            f"Activo: {symbol}\n"
            f"El *Shadow Journal* detectó anomalías estructurales (Autoencoder) o Alpha Decay (MDD).\n"
            f"🛑 *Trade Bloqueado.* El modelo entra en cuarentena silenciosa para proteger el capital institucional."
        )
        self.send_message(msg)

    def alert_mlops_resurrection(self, symbol: str):
        msg = (
            f"✅ *CUARENTENA MLOps LEVANTADA* ✅\n"
            f"Activo: {symbol}\n"
            f"El *Shadow Journal* evaluó los últimos 300 días y confirmó que la estadística volvió a la normalidad.\n"
            f"▶️ *Operaciones reactivadas.*"
        )
        self.send_message(msg)

    def alert_concept_drift(self, symbol: str):
        msg = (
            f"⚠️ *AVISO DE MANTENIMIENTO: CONCEPT DRIFT* ⚠️\n"
            f"Activo: {symbol}\n"
            f"El filtro de anomalías (Autoencoder) está envejeciendo. La mediana de errores recientes superó el límite P90 del entrenamiento.\n"
            f"👉 *Recomendación:* El bot seguirá operando, pero se sugiere correr el `portfolio_backtester.py` este fin de semana para refrescar y re-entrenar la estadística MLOps."
        )
        self.send_message(msg)

    def alert_max_hold_exit(self, symbol: str, max_hold: int):
        msg = (
            f"⏳ *CIERRE POR BARRERA VERTICAL (MAX HOLD)* ⏳\n"
            f"Activo: {symbol}\n"
            f"La posición abierta en *{symbol}* ha cumplido su límite de tiempo de `{max_hold}` días/velas sin alcanzar el Take Profit ni el Stop Loss.\n"
            f"✂️ *Acción:* Posición cerrada automáticamente a mercado para liberar capital y evitar costo de oportunidad."
        )
        self.send_message(msg)


