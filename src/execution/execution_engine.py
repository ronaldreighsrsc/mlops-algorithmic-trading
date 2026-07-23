import MetaTrader5 as mt5
import logging
from typing import Optional, Dict
from datetime import datetime


class ExecutionEngine:
    """
    Motor de ejecución para interactuar con MetaTrader 5 enviando órdenes y gestionando posiciones.
    """
    def __init__(self, connector):
        """
        :param connector: Instancia de MT5Connector ya autenticada.
        """
        self.connector = connector
        self.magic_number = 123456 # Identificador unico para las ordenes de nuestro bot

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene informacion especifica del simbolo (ticks, valor de lote, steps, etc).
        """
        if not self.connector.connected:
            return None
            
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logging.error(f"Simbolo {symbol} no encontrado o no visible en MT5.")
            return None
            
        if not sym_info.visible:
            logging.info(f"Haciendo visible el simbolo {symbol} en el Market Watch...")
            if not mt5.symbol_select(symbol, True):
                logging.error(f"Fallo al habilitar {symbol} en MT5.")
                return None
                
        return {
            'tick_size': sym_info.trade_tick_size,
            'tick_value': sym_info.trade_tick_value,
            'volume_step': sym_info.volume_step,
            'volume_min': sym_info.volume_min,
            'volume_max': sym_info.volume_max
        }

    def has_open_positions(self, symbol: str) -> bool:
        """
        Verifica si hay posiciones abiertas para nuestro magic number y simbolo.
        """
        if not self.connector.connected:
            return False
            
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
            
        # Filtramos por nuestro magic number
        bot_positions = [p for p in positions if p.magic == self.magic_number]
        return len(bot_positions) > 0

    def send_market_buy_order(self, symbol: str, volume: float, sl_price: float, tp_price: float) -> bool:
        """
        Envia una orden de compra a mercado con Stop Loss y Take Profit exactos.
        """
        if not self.connector.connected:
            return False
            
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return False
            
        point = sym_info.point
        ask_price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY,
            "price": ask_price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20, # Desviacion maxima de slippage permitida (en points)
            "magic": self.magic_number,
            "comment": "QuantBot Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel suele ser necesario en Darwinex
        }
        
        logging.info(f"Enviando orden BUY {volume} lotes en {symbol} a {ask_price} | SL: {sl_price} | TP: {tp_price}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Fallo envio de orden. Error code: {result.retcode} | Mensaje: {result.comment}")
            return False
            
        logging.info(f"Orden ejecutada con exito. Ticket: {result.order}")
        return True

    def send_market_sell_order(self, symbol: str, volume: float, sl_price: float, tp_price: float) -> bool:
        """
        Envia una orden de VENTA (Short) a mercado con Stop Loss y Take Profit.
        SL está ARRIBA del precio (si sube, pierdes) y TP está ABAJO (si baja, ganas).
        """
        if not self.connector.connected:
            return False
            
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return False
            
        bid_price = mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_SELL,
            "price": bid_price,
            "sl": float(sl_price),   # SL arriba del precio de entrada
            "tp": float(tp_price),   # TP abajo del precio de entrada
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "QuantBot Sell (Short)",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        logging.info(f"Enviando orden SELL (SHORT) {volume} lotes en {symbol} a {bid_price} | SL: {sl_price} | TP: {tp_price}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Fallo envio de orden SHORT. Error code: {result.retcode} | Mensaje: {result.comment}")
            return False
            
        logging.info(f"Orden SHORT ejecutada con exito. Ticket: {result.order}")
        return True

    def close_all_positions(self, symbol: str) -> bool:
        """
        Cierra todas las posiciones abiertas por el bot en el simbolo especificado
        (Util para boton de panico o Time Barrier).
        """
        if not self.connector.connected:
            return False
            
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return True
            
        bot_positions = [p for p in positions if p.magic == self.magic_number]
        success = True
        
        for pos in bot_positions:
            tick = mt5.symbol_info_tick(symbol)
            
            # Si es COMPRA, cerramos VENDIENDO al Bid
            if pos.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            # Si es VENTA, cerramos COMPRANDO al Ask
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "QuantBot Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Error cerrando posicion {pos.ticket}: {result.comment}")
                success = False
            else:
                logging.info(f"Posicion {pos.ticket} cerrada exitosamente.")
                
        return success

    def check_and_close_vertical_barrier(self, symbol: str, timeframe: int, max_hold: int = 10) -> bool:
        """
        Verifica si alguna posición abierta ha superado el límite de velas (Max Hold / Barrera Vertical)
        y la cierra a mercado. Retorna True si cerró al menos una posición.
        """
        if not self.connector.connected:
            return False
            
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return False
            
        bot_positions = [p for p in positions if p.magic == self.magic_number]
        closed_any = False
        
        for pos in bot_positions:
            pos_open_time = datetime.fromtimestamp(pos.time)
            
            # Obtener el número de velas transcurridas desde que se abrió el trade
            rates = mt5.copy_rates_range(symbol, timeframe, pos_open_time, datetime.now())
            if rates is not None:
                bars_held = len(rates) - 1 # Excluir la vela en formación actual
                if bars_held >= max_hold:
                    logging.info(f"[{symbol}] ⏳ BARRERA VERTICAL ALCANZADA: La posición {pos.ticket} lleva {bars_held} velas abiertas (Max Hold: {max_hold}). Cerrando a mercado...")
                    
                    tick = mt5.symbol_info_tick(symbol)
                    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": self.magic_number,
                        "comment": "QuantBot MaxHold Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"[{symbol}] Posición {pos.ticket} cerrada exitosamente por Barrera Vertical.")
                        closed_any = True
                    else:
                        logging.error(f"[{symbol}] Fallo al cerrar posición {pos.ticket} por Max Hold: {result.comment}")
                        
        return closed_any

