@echo off
:: Script de arranque del QuantBot - AWS Production (v2.0 Dual Mode)
title QuantBot v2.0 - Sistema de Trading Cuantitativo Institucional

:: Esperar 60 segundos para que MetaTrader 5 termine de cargar al inicio
timeout /t 60 /nobreak

:: Activar el entorno virtual y arrancar el bot en modo dual
cd /d C:\Users\Administrator\Desktop\quant-trading-bot
call venv\Scripts\activate.bat
python src\execution\main_bot_v2.py --mode dual --interval 5.0

:: Si el bot muere, esperar 30 segundos y reiniciar automaticamente
:restart
echo Bot detenido. Reiniciando en 30 segundos...
timeout /t 30 /nobreak
python src\execution\main_bot_v2.py --mode dual --interval 5.0
goto restart
