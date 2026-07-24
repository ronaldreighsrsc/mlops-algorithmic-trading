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

### 6. MLflow Experiment Tracking & Model Registry
Seguimiento automático de experimentos MLOps. Registra hiperparámetros, métricas estadísticas y financieras (Sharpe, Alpha, Win Rate, ROI), gráficos de equidad y artefactos de modelos. Permite comparar ejecuciones históricas y versionar campeones mediante una interfaz web interactiva accesible vía `mlflow ui`.

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

### 1. Extracción y Preprocesamiento de Datos Crudos (Multi-Timeframe)
```bash
python src/data_extractor.py
python src/main_preprocessing.py
```
*Se conecta a MT5 y Yahoo Finance para extraer velas desde el año 2000 en múltiples temporalidades descorrelacionadas (`D1`, `H4`, `H1`). Aplica FFD, EGARCH, Triple Barrera y alineamiento `ffill` de exógenas macro (VIX, DXY, Yield10Y, Macro Chile), manteniendo compatibilidad retroactiva por defecto con `D1`.*


### 2. El "Retuning" Maestro (Generar Obreros con Optuna)
```bash
python src/main_training.py
```
*El script más pesado. Ejecútalo 1 o 2 veces al año. Pone a competir a decenas de arquitecturas (XGBoost, RandomForest, LSTM, BiLSTM, ARIMA-LSTM, LSTM-RF) utilizando **Optimización Bayesiana (Optuna TPE Sampler + Purged CV Pruning)**. A diferencia de las búsquedas aleatorias tradicionales, Optuna aprende del historial de hiperparámetros y poda ejecuciones poco prometedoras rápidamente, reduciendo los tiempos de cómputo en un 40-60%. Emite archivos `.npy` con predicciones puras y `.pkl` con modelos entrenados en `results/`.*


### 3. Simulación Financiera y Entrenamiento MLOps (`portfolio_backtester.py`)
```bash
python src/evaluation/portfolio_backtester.py
```
*Este es el corazón analítico del sistema. Ejecuta el Torneo Financiero completo con **Kelly Dinámico** y **HRP**:*

#### 🔄 Modos de Ejecución (`fast_mode`):
- **Re-entrenamiento MLOps Anual (`fast_mode=False`):** (~1-2 horas). Entrena los modelos de detección de anomalías (HMM y LSTM Autoencoder) desde cero para cada combinación de activo/modelo/banco sobre los datos In-Sample. Se ejecuta **1 o 2 veces al año** y guarda los monitores entrenados en `results/mlops_monitors/`.
- **Evaluación Rápida (`fast_mode=True`):** (~2 minutos - **Modo por defecto**). Carga los Autoencoders y HMMs pre-entrenados desde `results/mlops_monitors/` en segundos para simular el capital en USD, calcular el **HRP** y generar los archivos de campeones.

> 💡 **Orden de Ejecución Recomendado:**
> 1. `main_training.py` (Genera predicciones `.npy` de los modelos base).
> 2. `portfolio_backtester.py` con `fast_mode=False` (Entrena y guarda los monitores MLOps).
> 3. `portfolio_backtester.py` con `fast_mode=True` o `backtester.py` (Evaluación diaria/semanal rápida).

### 3b. Backtester Rápido Independiente (`backtester.py`)
```bash
python src/evaluation/backtester.py
```
*Si solo quieres regenerar los reportes HTML, los gráficos de curva de equidad o exportar la configuración del campeón sin pasar por la simulación de billetera en USD, corre el backtester independiente (`fast_mode=True` por defecto).*

### 3c. Auditoría de Robustez y PBO (`cpcv_auditor.py`)
```bash
python src/evaluation/cpcv_auditor.py
```
*Certifica matemáticamente que la estrategia del campeón NO fue fruto del sobreajuste (Overfitting) ni de la suerte. Aplica **Combinatorial Purged Cross-Validation (CPCV)** dividiendo la historia en combinaciones de caminos cruzados ($\binom{6}{2} = 15$ caminos) y calcula la **Probability of Backtest Overfitting (PBO)**. Genera el gráfico `cpcv_sharpe_distribution_{activo}.png` en `results/` y registra la distribución en MLflow.*


### 4. Puesta en Producción (Live Trading)
```bash
python src/execution/main_bot.py
```
*El ciclo de vida final. El bot carga al campeón desde su `.json` y extrae los pesos MLOps (`.keras`, `.pkl`). Ejecuta su **Shadow Journal** evaluando los últimos 300 días para auto-diagnosticar su salud (Cuarentena / Concept Drift). Si el diagnóstico es exitoso (`✅ Shadow Journal OK`), procesa la última vela de hoy, gestiona la Barrera Vertical (`Max Hold`), dispara la orden de compra/venta a MT5 y envía notificaciones por Telegram con la calculadora dual (Lotes MT5 y Trading Power exacto para ejecución manual en Quantfury).*

### 4b. Empaquetar para AWS (Generar `bot_production.zip`)
```bash
python export_to_aws.py
```
*Empaqueta de forma inteligente solo los modelos campeones activos, sus monitores MLOps (`.keras`, `.pkl`), la matriz `hrp_weights.json`, el código fuente `src/` y las dependencias.*

---

## 📅 Calendario de Mantenimiento MLOps (Cuándo ejecutar qué script)

Para no confundir qué script debe correr con qué frecuencia ni qué parámetro usar, sigue este cuadro operativo:

| Fase MLOps | Frecuencia Recomendada | Script a Ejecutar | Parámetro Clave | Qué hace / Qué genera |
|---|---|---|---|---|
| **1. Refresco de Datos** | Mensual | `python src/data_extractor.py`<br>`python src/main_preprocessing.py` | N/A | Descarga velas recientes y actualiza `data/processed/*.csv` |
| **2. Re-entrenamiento de Modelos IA** | 1 o 2 veces al año | `python src/main_training.py` | N/A | Re-entrena XGBoost, BiLSTM, ARIMA-LSTM sobre nuevos datos. Genera `.pkl` y `.npy`. |
| **3. Entrenamiento Monitores MLOps** | 1 o 2 veces al año *(tras Paso 2)* | `python src/evaluation/portfolio_backtester.py` | `fast_mode=False`<br>*(~1-2 horas)* | Entrena los detectores HMM (Markov) y LSTM Autoencoders desde cero en `results/mlops_monitors/`. |
| **4. Auditoría PBO & CPCV** | Trimestral / Tras Paso 2 | `python src/evaluation/cpcv_auditor.py` | N/A | Evalúa $\binom{6}{2}=15$ caminos cruzados y certifica PBO < 5%. |
| **5. Rebalanceo de Pesos HRP** | Mensual (ej. el 1º de cada mes) | `python src/evaluation/portfolio_backtester.py` | `fast_mode=True`<br>*(~2 minutos)* | Carga monitores pre-entrenados, recalcula la matriz HRP sobre datos recientes y actualiza `hrp_weights.json`. |
| **6. Empaquetado AWS** | Tras cada Paso 2 o 5 | `python export_to_aws.py` | N/A | Genera el archivo `bot_production.zip` listo para desplegar. |
| **7. Ejecución 24/7** | Continuo en AWS | `python src/execution/main_bot.py` | N/A | Corre en vivo en el servidor, descarga velas del día, pasa por el Shadow Journal y opera. |


---

> [!TIP]
> Corre este comando cada vez que hagas cambios en el código o re-entrenes los modelos. Luego sube el `bot_production.zip` a tu instancia EC2, descomprímelo y reinicia el bot.

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

## 🧪 Tests Unitarios (MLOps)

El proyecto incluye **26 pruebas unitarias** con `pytest` que validan las matemáticas críticas del bot para prevenir bugs silenciosos que podrían quemar la cuenta:

```bash
python -m pytest tests/ -v
```

| Módulo | Tests | Qué protege |
|--------|-------|-------------|
| `test_risk_manager.py` | 6 | Position Sizing, Kelly, barreras TP/SL |
| `test_triple_barrier.py` | 6 | Etiquetado correcto (bull/crash/flat) |
| `test_ffd.py` | 6 | Estacionariedad, memoria, columnas protegidas |
| `test_egarch.py` | 4 | Volatilidad positiva, cap 5%, cadena de fallback |
| `test_technical_features.py` | 4 | RSI [0,100], ATR > 0, fail-fast |

> [!TIP]
> Corre `pytest` después de cualquier cambio en los módulos de preprocesamiento o riesgo para asegurar que no introdujiste un bug silencioso.

## 📊 MLflow Dashboard (Experiment Tracking & Model Registry)

El sistema integra **MLflow** para registrar automáticamente cada experimento de entrenamiento, torneo de backtest y simulación financiera de portafolio.

### Iniciar la Interfaz Web Local
```bash
mlflow ui
```
Abre tu navegador en `http://127.0.0.1:5000` para visualizar:
- **Training Experiments:** Hiperparámetros, cantidad de predicciones OOS y artefactos `.pkl` / `.keras` por cada combinación modelo/banco.
- **Tournament Runs:** Alpha, Win Rate, Sharpe, Sortino, Calmar, Max Drawdown y Deflated Sharpe Ratio (DSR) de cada candidato evaluado.
- **Portfolio Evaluation:** ROI Total, ROI Anualizado, Capital Final, gráficos de equidad y configuración del Campeón exportado para producción.

## ⚙️ Configuración del Entorno

1. **Python 3.12 (64-bits)** requerido.
2. Crea el archivo `.env` en la raíz con credenciales de MT5 (Darwinex-Demo) y Tokens de Telegram (Opcional).
3. **Entorno Virtual**: Es obligatorio instalar las dependencias aisladas para evitar conflictos de versiones con Scipy y TensorFlow.
   - **Crear entorno:** `python -m venv venv`
   - **Activar entorno (Windows):** `venv\Scripts\activate`
   - **Activar entorno (Mac/Linux):** `source venv/bin/activate`
   - **Instalar dependencias:** `pip install -r requirements.txt`