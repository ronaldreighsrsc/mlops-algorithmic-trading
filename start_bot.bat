@echo off
:: Script de arranque del QuantBot - AWS Production
title QuantBot - Sistema de Trading Cuantitativo

:: Esperar 60 segundos para que MetaTrader 5 termine de cargar al inicio
timeout /t 60 /nobreak

:: Activar el entorno virtual y arrancar el bot
cd /d C:\Users\Administrator\Desktop\quant-trading-bot
call venv\Scripts\activate.bat
python src\execution\main_bot.py

:: Si el bot muere, esperar 30 segundos y reiniciar automaticamente
:restart
echo Bot detenido. Reiniciando en 30 segundos...
timeout /t 30 /nobreak
python src\execution\main_bot.py
goto restart
