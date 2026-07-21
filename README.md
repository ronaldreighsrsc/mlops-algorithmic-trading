# Institutional Quant Trading Bot (MT5 & Darwinex)

Este repositorio contiene un sistema de trading algorítmico cuantitativo de grado institucional, diseñado para operar de manera automatizada a través de MetaTrader 5 (MT5), específicamente enfocado en el broker Darwinex.

El sistema emplea el marco teórico avanzado de **Marcos López de Prado** (Advances in Financial Machine Learning), utilizando un ensamble de modelos tradicionales (Random Forest, XGBoost), redes neuronales profundas (LSTM, BiLSTM), arquitecturas híbridas avanzadas (ARIMA-LSTM, LSTM-RF), control de riesgo estocástico, detectores de anomalías MLOps y transformaciones matemáticas rigurosas para operar en temporalidades diarias (D1).

## 🧠 Arquitectura Core

El bot está dividido en 5 pilares fundamentales:

1. **Fractional Differencing (FFD)**: Estacionariedad preservando la memoria. Transforma las series de precios conservando el máximo de información (Test de Dickey-Fuller).
2. **Triple Barrier Method**: Etiquetado de datos con meta-labeling. Define 3 barreras dinámicas: Take Profit y Stop Loss basados en la volatilidad condicional diaria (EGARCH) + Barrera Vertical (`Max Hold` de 10 días) que liquida a mercado la posición si no alcanza los objetivos en tiempo.
3. **Volatilidad Condicional (EGARCH)**: Ajusta el ancho de las barreras de TP/SL diariamente según la volatilidad proyectada del mercado.
4. **Machine Learning Predictivo**: Redes Neuronales (LSTM, BiLSTM), Gradient Boosting (XGBoost) e Híbridos (ARIMA-LSTM) entrenados con Validación Cruzada Purgada y Embargo (Purged K-Fold) + Walk-Forward Optimization para prevenir fuga de datos temporales.
5. **Gestión de Riesgo (Kelly Dinámico)**: Escala el lote de inversión dinámicamente (0.5x, 1.0x, 2.0x) según la fuerza de la probabilidad estadística predicha.

---

## 🛡️ Arquitectura Institucional MLOps (Gestión de Riesgos)

En la versión actual, el sistema implementa una arquitectura robusta de control de estado ("Health Check") del modelo matemático para prevenir colapsos de capital:

### 1. Cuarentena por Anomalías Microestructurales (Soft-Stop)
Utilizamos un **LSTM Autoencoder** pre-entrenado que funciona como un *Hybrid Risk Monitor*. Escucha la distribución de los últimos 10-30 trades. Si detecta una perturbación matemática aguda (el error de reconstrucción supera el P99) que provoca más de 3% de pérdida rápida, **el modelo entra en Cuarentena de 60 días**. Pasa a efectivo (Cash) para permitir que el proceso de *Walk-Forward* re-aprenda el nuevo régimen de mercado. A los 60 días, resucita.

### 2. Alpha Decay y Muerte Permanente (Hard Kill-Switch Dinámico)
Si el mercado sufre un cambio fundamental incorregible, la estrategia sufrirá el llamado "Alpha Decay". El sistema calcula estadísticamente el Riesgo de Ruina basándose en tu `RIESGO_PCT` (por defecto 2.5%). Si la estrategia **supera un Maximum Drawdown (MDD) equivalente a 10 pérdidas máximas consecutivas (ej. -25% para un riesgo del 2.5%)** desde su pico de capital histórico, es declarada matemáticamente muerta (`💀 MUERTO`). **No resucita nunca más.** Queda vetada del entorno de Producción hasta que se corra un Hyperparameter Retuning masivo.

> [!NOTE] 
> **Filosofía Institucional del Riesgo:** El multiplicador del Kill-Switch se mantiene unificado (x10) para todos los modelos, en lugar de calcularse dinámicamente según el Win Rate *In-Sample* de cada uno. Esto evita el *Overfitting* de reglas de riesgo y previene que un modelo ineficiente se auto-asigne límites de pérdida enormes. En la arquitectura cuantitativa seria: **El inversor define el límite máximo de dolor (Capital Tolerance), no el modelo estadístico.**

### 3. Detector de Vejez (Concept Drift Detector)
La estadística del mercado envejece. El sistema calcula constantemente la Mediana del Error de Reconstrucción de los últimos 30 trades en producción y lo compara contra la frontera P90 del entrenamiento In-Sample. Si la mediana reciente rompe esta frontera, el sistema avisa que **el modelo está estadísticamente obsoleto (Concept Drift)** y requiere re-entrenamiento (Retuning) urgente, sin necesidad de esperar a sufrir pérdidas severas.

### 4. Shadow Journal (Diario Sin Estado en Producción)
Para alimentar los detectores de MLOps en el día a día sin depender de bases de datos o archivos `.csv` corruptibles, el bot en vivo utiliza una arquitectura **Stateless**. Cada mañana descarga los últimos 300 días de historial, predice las probabilidades "al vuelo" y simula internamente las operaciones recientes usando el `TripleBarrierBacktester`. El resultado de esta simulación "sombra" le permite saber instantáneamente si está en racha perdedora y auto-bloquearse (activar Cuarentena) antes de lanzar la orden del día.

### 5. Hierarchical Risk Parity (HRP)
Implementación nativa del algoritmo de asignación de capital de Marcos López de Prado. En lugar de utilizar la inestable optimización de Markowitz, el sistema agrupa los activos según su correlación histórica mediante *Clustering Jerárquico*. El simulador global calcula los pesos ideales, y el bot en Producción lee esta matriz resultante (`hrp_weights.json`), reduciendo dinámicamente el presupuesto (lotes) de aquellos bots que estén altamente correlacionados entre sí para maximizar la diversificación real.

---

## 📁 Estructura del Proyecto

```text
quant-trading-bot/
 |-- src/
 |   |-- preprocessing/
 |   |   |-- stationarity.py       # Transformación FFD y Test ADF
 |   |   |-- triple_barrier.py     # Etiquetado de meta-labeling
 |   |   |-- volatility.py         # Cálculo de EGARCH
 |   |   |-- technical_features.py # Indicadores Técnicos Clásicos
 |   |   |-- auditor.py            # Auditoría Matemática de Datasets (DataAuditor)
 |   |-- models/
 |   |   |-- anomaly_detector.py   # LSTM Autoencoder y HMMRegimeDetector
 |   |   |-- bilstm_model.py       # BiLSTM Core
 |   |   |-- xgb_model.py          # XGBoost Core
 |   |   |-- arima_lstm.py         # Híbrido ARIMA + LSTM
 |   |   |-- lstm_rf.py            # Híbrido LSTM + Random Forest
 |   |-- evaluation/
 |   |   |-- backtester.py         # TripleBarrierBacktester (fast_mode: genera reportes y JSON)
 |   |   |-- portfolio_backtester.py # Simulador Financiero en USD (entrena MLOps + Kelly + HRP)
 |   |   |-- hrp_optimizer.py      # Hierarchical Risk Parity (López de Prado)
 |   |-- execution/
 |   |   |-- main_bot.py           # Live Trading Bot MT5 (Carga Filtros MLOps y Predice)
 |   |   |-- risk_manager.py       # Monitor Híbrido de Riesgo (HMM + Autoencoder)
 |   |-- main_training.py          # Pipeline maestro de Retuning de Obreros (Walk-Forward)
 |   |-- main_preprocessing.py     # Pipeline maestro de Preprocesamiento (FFD, EGARCH, Triple Barrera)
 |   |-- data_extractor.py         # Conexión a MT5 y Yahoo Finance
 |-- results/                      # ⚠️ Ignorado por .gitignore (Protección de Alpha)
 |   |-- saved_models/             # Modelos ML/DL entrenados (.pkl)
 |   |-- mlops_monitors/           # HMM y Autoencoder pre-entrenados por portfolio_backtester
 |   |-- *.npy                     # Probabilidades In-Sample y Out-of-Sample
 |   |-- campeon_*.json            # Configuración del mejor modelo para Producción
 |-- requirements.txt              
 |-- .env                          
```

## 🔄 Pipeline End-to-End (Cómo Usar el Proyecto)

El sistema está diseñado para fluir de manera secuencial. Cada paso depende del anterior.

### 1. Extracción y Preprocesamiento de Datos Crudos
```bash
python src/data_extractor.py
python src/main_preprocessing.py
```
*Se conecta a MT5 y Yahoo Finance para extraer desde el año 2000. Aplica FFD, EGARCH, Triple Barrera y guarda todo en `data/processed/`.*

### 2. El "Retuning" Maestro (Generar Obreros)
```bash
python src/main_training.py
```
*El script más pesado. Ejecútalo 1 o 2 veces al año. Pone a competir a decenas de arquitecturas (XGBoost, LSTM, BiLSTM, ARIMA-LSTM, LSTM-RF) con diferentes bancos de datos. Realiza Grid Search, Purged K-Fold y genera simulaciones de Walk-Forward. Emite archivos `.npy` con predicciones puras y `.pkl` con modelos entrenados en `results/`.*

### 3. Simulación Financiera Completa (El Script Pesado)
```bash
python src/evaluation/portfolio_backtester.py
```
*Este es el corazón analítico del sistema. Ejecuta el Torneo Financiero completo:*
- *Lee las predicciones `.npy` de los obreros y **entrena a los Gerentes de Riesgo (HMM y LSTM Autoencoder)** sobre los datos In-Sample.*
- *Guarda todos los modelos MLOps entrenados en `results/mlops_monitors/` para reutilización rápida.*
- *Aplica las reglas de **Cuarentena (60 días)** y **Muerte por MDD (15%)** al Out-of-Sample.*
- *Simula el capital real en USD con **Kelly Dinámico** y genera gráficos de rendimiento.*
- *Ejecuta la optimización **HRP (Hierarchical Risk Parity)** multi-activo y exporta los pesos a `hrp_weights.json`.*
- *Corona al mejor modelo como `campeon_{activo}.json` con sus filtros MLOps listos para Producción.*

> ⏱️ **Duración estimada:** ~1-2 horas (entrena HMM + Autoencoder para cada combinación modelo/banco).

### 3b. Backtester Rápido y Reportes (`fast_mode`)
Si solo quieres regenerar los reportes, el JSON de producción o revisar los resultados sin tener que esperar 2 horas de entrenamiento, corre el backtester independiente.
Por defecto usa `fast_mode=True`, lo que significa que carga los Autoencoders y modelos pre-entrenados desde `results/mlops_monitors/` en segundos:

```bash
python src/evaluation/backtester.py
```

### 4. Puesta en Producción (Live Trading)
```bash
python src/execution/main_bot.py
```
*El ciclo de vida final. El bot carga al campeón desde su `.json` y extrae los pesos MLOps (`.keras`, `.pkl`). Ejecuta su **Shadow Journal** evaluando los últimos 300 días para auto-diagnosticar su salud (Cuarentena / Concept Drift). Si el diagnóstico es exitoso (`✅ Shadow Journal OK`), procesa la última vela de hoy, gestiona la Barrera Vertical (`Max Hold`), dispara la orden de compra/venta a MT5 y envía notificaciones por Telegram con la calculadora dual (Lotes MT5 y Trading Power exacto para ejecución manual en Quantfury).*

### 5. Automatización en Servidor AWS / VPS (Recomendado)
Para que el bot corra 24/7 y sobreviva a reinicios automáticos de AWS (parches de Windows), **NO** se debe usar un arranque en modo servicio ("Session 0"), ya que MetaTrader 5 requiere entorno gráfico (GUI) para funcionar sin crashear. 

Sigue estos 2 pasos para configurarlo correctamente de manera institucional:

**Paso 1: Activar Auto-Login en Windows Server**
1. Abre el Símbolo del Sistema (CMD) como Administrador y ejecuta este comando para destrabar la configuración oculta de Windows:
   `reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device" /v DevicePasswordLessBuildVersion /t REG_DWORD /d 0 /f`
2. Presiona `Win + R`, escribe `netplwiz` y dale a Enter.
3. Desmarca la casilla *"Users must enter a user name and password to use this computer"*.
4. Dale a Aplicar, introduce tu contraseña de Administrator dos veces y acepta. (Ahora el servidor iniciará sesión y cargará el escritorio automáticamente al encender).

**Paso 2: Crear la Tarea Programada (Al Iniciar Sesión)**
Abre CMD como Administrador y crea la tarea para que lance el archivo `.bat` justo cuando el escritorio cargue:
```cmd
schtasks /create /tn "QuantBot_Trading" /tr "C:\Users\Administrator\Desktop\quant-trading-bot\start_bot.bat" /sc onlogon /ru "Administrator" /rl highest /f
```
> [!IMPORTANT]
> Observa el parámetro `/sc onlogon` (Al iniciar sesión). Usar `/sc onstart` (Al encender) ejecutará el bot oculto en el fondo, impidiéndote ver los gráficos de MT5 e inestabilizando la conexión Python-MT5.

## ⚙️ Configuración del Entorno

1. **Python 3.12 (64-bits)** requerido.
2. Crea el archivo `.env` en la raíz con credenciales de MT5 (Darwinex-Demo) y Tokens de Telegram (Opcional).
3. **Entorno Virtual**: Es obligatorio instalar las dependencias aisladas para evitar conflictos de versiones con Scipy y TensorFlow.
   - **Crear entorno:** `python -m venv venv`
   - **Activar entorno (Windows):** `venv\Scripts\activate`
   - **Activar entorno (Mac/Linux):** `source venv/bin/activate`
   - **Instalar dependencias:** `pip install -r requirements.txt`