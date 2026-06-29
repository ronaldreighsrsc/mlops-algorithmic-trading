# Institutional Quant Trading Bot (MT5 & Darwinex)

Este repositorio contiene un sistema de trading algorítmico cuantitativo de grado institucional, diseñado para operar de manera automatizada a través de MetaTrader 5 (MT5), específicamente enfocado en el broker Darwinex.

El sistema emplea el marco teórico avanzado de **Marcos López de Prado** (Advances in Financial Machine Learning), utilizando un ensamble de modelos tradicionales (Random Forest, XGBoost), redes neuronales profundas (LSTM, BiLSTM), arquitecturas híbridas avanzadas (ARIMA-LSTM, LSTM-RF), control de riesgo estocástico, detectores de anomalías MLOps y transformaciones matemáticas rigurosas para operar en temporalidades diarias (D1).

## 🧠 Arquitectura Core

El bot está dividido en 5 pilares fundamentales:

1. **Fractional Differencing (FFD)**: Estacionariedad preservando la memoria. Transforma las series de precios conservando el máximo de información (Test de Dickey-Fuller).
2. **Triple Barrier Method**: Etiquetado de datos con meta-labeling. Define Take Profit y Stop Loss dinámicos basados en la volatilidad condicional diaria.
3. **Volatilidad Condicional (EGARCH)**: Ajusta el ancho de las barreras de TP/SL diariamente según la volatilidad proyectada del mercado.
4. **Machine Learning Predictivo**: Redes Neuronales (LSTM, BiLSTM), Gradient Boosting (XGBoost) e Híbridos (ARIMA-LSTM) entrenados con Validación Cruzada Purgada y Embargo (Purged K-Fold) + Walk-Forward Optimization para prevenir fuga de datos temporales.
5. **Gestión de Riesgo (Kelly Dinámico)**: Escala el lote de inversión dinámicamente (0.5x, 1.0x, 2.0x) según la fuerza de la probabilidad estadística predicha.

---

## 🛡️ Arquitectura Institucional MLOps (Gestión de Riesgos)

En la versión actual, el sistema implementa una arquitectura robusta de control de estado ("Health Check") del modelo matemático para prevenir colapsos de capital:

### 1. Cuarentena por Anomalías Microestructurales (Soft-Stop)
Utilizamos un **LSTM Autoencoder** pre-entrenado que funciona como un *Hybrid Risk Monitor*. Escucha la distribución de los últimos 10-30 trades. Si detecta una perturbación matemática aguda (el error de reconstrucción supera el P99) que provoca más de 3% de pérdida rápida, **el modelo entra en Cuarentena de 60 días**. Pasa a efectivo (Cash) para permitir que el proceso de *Walk-Forward* re-aprenda el nuevo régimen de mercado. A los 60 días, resucita.

### 2. Alpha Decay y Muerte Permanente (Hard Kill-Switch)
Si el mercado sufre un cambio fundamental incorregible, la estrategia sufrirá el llamado "Alpha Decay". Si la estrategia **supera un Maximum Drawdown (MDD) de -15% desde su pico de capital histórico**, es declarada matemáticamente muerta (`💀 MUERTO`). **No resucita nunca más.** Queda vetada del entorno de Producción hasta que se corra un Hyperparameter Retuning masivo.

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
> 💡 **Nota sobre `fast_mode`:** En este paso se usan dos pases. Primero simula cada activo individualmente entrenando sus escudos MLOps desde cero (`fast_mode=False`) y guardándolos en disco. Luego, junta todos en el simulador global HRP usando los escudos guardados (`fast_mode=True`) para evitar duplicar el tiempo de entrenamiento de la IA.

### 3b. Regenerar Reportes Rápidamente (Modo Fast)
```bash
python src/evaluation/backtester.py
```
*Ejecuta el mismo Torneo pero en **modo rápido (`fast_mode=True`)**. En lugar de re-entrenar los HMM y Autoencoders desde cero, los **carga desde disco** (`results/mlops_monitors/`). Genera el reporte HTML profesional, gráficos de Equity y el JSON de producción en segundos. Requiere haber corrido `portfolio_backtester.py` al menos una vez.*

### 4. Puesta en Producción (Live Trading)
```bash
python src/execution/main_bot.py
```
*El ciclo de vida final. El bot carga al campeón desde su `.json` y extrae los pesos MLOps (`.keras`, `.pkl`). Ejecuta su **Shadow Journal** evaluando los últimos 300 días para auto-diagnosticar su salud (Cuarentena / Concept Drift). Si el diagnóstico es exitoso (`✅ Shadow Journal OK`), procesa la última vela de hoy y dispara la orden de compra/venta a MT5 (o alerta Telegram para ECH) usando dimensionamiento Kelly.*

## ⚙️ Configuración del Entorno

1. **Python 3.12 (64-bits)** requerido.
2. Crea el archivo `.env` en la raíz con credenciales de MT5 (Darwinex-Demo) y Tokens de Telegram (Opcional).
3. **Entorno Virtual**: Es obligatorio instalar las dependencias aisladas para evitar conflictos de versiones con Scipy y TensorFlow.
   - **Crear entorno:** `python -m venv venv`
   - **Activar entorno (Windows):** `venv\Scripts\activate`
   - **Activar entorno (Mac/Linux):** `source venv/bin/activate`
   - **Instalar dependencias:** `pip install -r requirements.txt`