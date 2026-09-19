# 📈 Autonomous Institutional Quantitative Trading Infrastructure
## AlphaEdge Sentinel v2.0 — MetaTrader 5 & Darwinex Enterprise Edition
### Marcos López de Prado Framework | MLOps Drift & Regime Governance | Ultra-Low Latency (< 15 ms)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![MetaTrader5](https://img.shields.io/badge/MetaTrader_5-Native_Bridge-green?style=for-the-badge)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-C++_BLAS_<1.8ms-005ced?style=for-the-badge&logo=onnx)
![VPS Footprint](https://img.shields.io/badge/VPS_RAM-<130MB_Zero_OOM-brightgreen?style=for-the-badge)
![Validation](https://img.shields.io/badge/CPCV-PBO_<5%25-orange?style=for-the-badge)
![Theory](https://img.shields.io/badge/López_de_Prado-FFD_&_HRP-purple?style=for-the-badge)
![Persistence](https://img.shields.io/badge/SQLite_WAL-ACID_In--Process-003B57?style=for-the-badge&logo=sqlite)
![Regimes](https://img.shields.io/badge/HMM_3--State-Bull_Bear_Choppy-red?style=for-the-badge)
![Pytest](https://img.shields.io/badge/Tests-63%2F63_Passing_100%25-success?style=for-the-badge&logo=pytest)
![MLflow](https://img.shields.io/badge/MLflow-Model_Registry-0194e2?style=for-the-badge&logo=mlflow)

Infraestructura de trading cuantitativo algorítmico automatizado de grado **Hedge Fund / Prop Trading Desk**, diseñada para operar de manera autónoma en **MetaTrader 5 (MT5)** y optimizada para el broker ECN **Darwinex**. El sistema transpone los principios de ingeniería de sistemas distribuidos, tolerancia a fallos bancarios y resiliencia edge hacia una arquitectura de ejecución de ultra-alta disponibilidad y baja latencia.

---

## 🏛️ La Trilogía de Misión Crítica Transversal

Existe una estricta simetría matemática y operacional entre los tres sistemas centrales del portafolio:
1. **Detección de Fraude Bancario (Banco Bci / CMF):** Autorizaciones síncronas en tiempo real (< 30 ms) vs. investigación forense y reportes ROS asíncronos.
2. **Resiliencia IoT & Edge (OmniEdge Sentinel):** Decisión de handover de red (< 5 ms en RAM) vs. persistencia segura en memoria flash sin bloqueo.
3. **Trading Cuantitativo (AlphaEdge Sentinel v2.0):** Envío de órdenes a mercado (< 15 ms) vs. reentrenamiento diario, Shadow Journal y análisis macroeconómico.

| Dimensión de Ingeniería | Sistema Bancario (Bci / Fraude) | Sistema Edge IoT (OmniEdge) | Sistema Quant Bot (v2.0) |
| :--- | :--- | :--- | :--- |
| **1. Restricción Temporal (SLA)** | Switch Transaccional (< 30 ms) | Handover Wi-Fi (< 800 ms) | **Tick-to-Order MT5 (< 15 ms)** |
| **2. Función de Pérdida / Costo** | Costo Asimétrico Ley 21.234 (40:1) | Penalización de Desconexión $C_{\text{switch}}$ | **Fricción Microestructural (Spread + Slippage + Swap)** |
| **3. Restricción de Cómputo** | Microservicios Cloud / Container | Memoria Flash SD / RAM < 120 MB | **VPS Trading 1-2 GB RAM (< 130 MB, Zero OOM)** |
| **4. Detección de Deriva (Drift)** | Population Stability Index (PSI) | Test Kolmogorov-Smirnov RF | **LSTM Autoencoder (P90/P99) + KS-Test 50 barras** |
| **5. Agente Inteligente / Explicabilidad** | Agente ROS CMF (Tipologías UAF) | Agente RCA Falla Red (IEEE 802.11) | **Agente Macro-Forense Pre-News & RCA Post-Trade** |
| **6. Persistencia y Caché** | Redis In-Memory + Delta Lake | Ring Buffer RAM + SQLite Batch | **Caching RAM + SQLite WAL In-Process (`TradeVault`)** |

---

## 🏗️ Arquitectura Desacoplada en Doble Motor (SLA < 15 ms vs. Async Worker)

En la operativa institucional real, calcular simulaciones pesadas de 300 días o reajustar gradientes neuronales en la ruta crítica de mercado destruye la ejecución provocando **deslizamientos severos (slippage)** y pérdidas de ventanas de liquidez. La versión **v2.0** desacopla completamente el sistema en dos motores autónomos sincronizados de forma no bloqueante a través de **SQLite Write-Ahead Logging (WAL)**:

```
═══════════════════════════════════════════════════════════════════════════════════════════════════
                   ARQUITECTURA INSTITUCIONAL DESACOPLADA (ALPHAEDGE SENTINEL v2.0)
═══════════════════════════════════════════════════════════════════════════════════════════════════

 [PROCESO 1: EXECUTION ENGINE GATEWAY (SLA < 15 ms)]
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MT5 New Tick / Bar Event                                                                    │
 │       │                                                                                     │
 │       ▼                                                                                     │
 │ CAPA 1: Hard Risk Kill-Switch en RAM ──► (Drawdown Diario > 2.5%?) ──► [ABORT IMMEDIATE]   │
 │       │ (Aprobado)                                                                          │
 │       ▼                                                                                     │
 │ CAPA 2: Estado de Cuarentena O(1) en TradeVault WAL ─────────────────► [ABORT CUARENTENA]  │
 │       │ (Saludable)                                                                         │
 │       ▼                                                                                     │
 │ CAPA 3: Pre-News Macro Hazard Guard (-30m / +15m FOMC, CPI, NFP) ───► [FREEZE PRE-NEWS]    │
 │       │ (Mercado Seguro)                                                                    │
 │       ▼                                                                                     │
 │ CAPA 4: Inferencia C++ BLAS con ONNX Runtime (< 1.8 ms) ────────────► Probabilidad P(TP)   │
 │       │                                                                                     │
 │       ▼                                                                                     │
 │ CAPA 5: Cost-Sensitive Gatekeeper ──► E[U] >= 2.5 * Fricción? ──────► [RECHAZAR POR COSTO] │
 │       │ (Aprobado)                                                                          │
 │       ▼                                                                                     │
 │ CAPA 6: Kelly Sizing Dinámico Modulado por HMM 3-Estados & KS-Drift                         │
 │       │                                                                                     │
 │       ▼                                                                                     │
 │ CAPA 7: MT5 Order Send (< 10 ms) & Registro Transaccional en SQLite WAL (`TradeVault`)      │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                ▲
                                │ Sincronización Asíncrona ACID (SQLite WAL `trading_vault.db`)
                                │ (system_state: cuarentena, drift, pesos HRP, heartbeat)
 [PROCESO 2: BACKGROUND ANALYTICS & MLOps DAEMON]
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ • Simulación periódica de Shadow Journal de 300 días                                        │
 │ • Monitoreo de Error de Reconstrucción LSTM Autoencoder (P90 Drift vs. P99 Kill-Switch)     │
 │ • Auto-Rebalanceo Mensual de Portafolio HRP (Shrinkage Bayesiano λ = 0.85)                  │
 │ • Recalibración Continua Bayesiana de Monte Carlo MDD cada 30 trades reales                 │
 │ • Verificación de Integridad Criptográfica de Modelos ONNX mediante Checksums SHA-256      │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detalle Técnico de los 6 Pilares Institucionales v2.0

### 1. Desacoplamiento Estricto de SLAs (`src/execution/`)
- **`execution_gateway.py` (Proceso 1):** Bucle de ejecución de alta velocidad con SLA garantizado inferior a **15 milisegundos**. No ejecuta reentrenamientos ni cálculos de 300 días; consulta el estado de salud del sistema en tiempo $\mathcal{O}(1)$ desde la memoria/WAL.
- **`analytics_daemon.py` (Proceso 2):** Proceso en segundo plano que se ejecuta fuera de los horarios críticos de apertura (o en hilo asíncrono no bloqueante), procesando el Shadow Journal, calibrando la volatilidad EGARCH y actualizando los pesos de portafolio HRP.
- **`main_bot_v2.py`:** Orquestador maestro que permite lanzar el sistema en 3 modalidades:
  - `--mode dual`: Modo unificado de grado producción (Gateway en hilo principal + Daemon MLOps asíncrono).
  - `--mode gateway`: Exclusivamente el motor de ejecución en vivo (ideal para VPS de ultra-bajos recursos).
  - `--mode daemon`: Exclusivamente el trabajador analítico MLOps.

### 2. Inferencia Ultra-Eficiente con ONNX Runtime (< 130 MB RAM, < 1.8 ms)
- **`onnx_inference_engine.py`:** Sustituye el grafo pesado de TensorFlow/Keras en producción por **ONNX Runtime (C++ BLAS)**:
  - Opciones de sesión optimizadas para VPS de 1 socket: `intra_op_num_threads=2`, `execution_mode=ORT_SEQUENTIAL`, `graph_optimization_level=ORT_ENABLE_ALL`.
  - Inferencia con memoria C-continua `np.ascontiguousarray` en `float32`.
  - **Reducción de RAM:** Pasa de ~1,400 MB a **menos de 130 MB**, eliminando el riesgo de Out-of-Memory (OOM) en instancias cloud económicas de 1 GB.
  - **Latencia:** Pasa de 50-80 ms a **< 1.8 ms** por pasada.
- **`onnx_exporter.py` & `export_champions_to_onnx.py`:** Herramientas de exportación con gobernanza criptográfica: calcula y valida el hash **SHA-256** de cada modelo frente al manifiesto inmutable `onnx_manifest.json`.

### 3. Filtro Microestructural Sensible al Costo (`cost_sensitive_gatekeeper.py`)
En el trading real institucional con brokers ECN (Darwinex), los modelos que ignoran la microestructura son destruidos por el ensanchamiento de spreads y el deslizamiento estocástico. El sistema implementa una **Función de Utilidad Neta Esperada**:

$$\mathbb{E}[U_t] = \hat{P}(\text{TP}) \cdot R_{TP} - (1 - \hat{P}(\text{TP})) \cdot |R_{SL}| - \mathcal{C}_{\text{fricción}}(t)$$

Donde el costo total de fricción $\mathcal{C}_{\text{fricción}}(t)$ se descompone estocásticamente en tiempo real como:

$$\mathcal{C}_{\text{fricción}}(t) = \text{Spread}_t + \underbrace{\left(0.20 \cdot \text{Spread}_t + 0.05 \cdot \text{ATR}_t \cdot \sqrt{\frac{V_t}{\bar{V}_t}}\right)}_{\text{Slippage Estocástico}} + \underbrace{\text{Swap Diario} \times \mathbb{E}[T_{\text{holding}}]}_{\text{Costo de Financiamiento}}$$

**Regla de Hurdle Rate Institucional:**
$$\text{Si } \frac{\mathbb{E}[U_t]}{\mathcal{C}_{\text{fricción}}(t)} < 2.5 \implies \textbf{RECHAZAR ORDEN (Edge Insuficiente)}$$

Si el retorno esperado bruto no supera al menos **2.5 veces el costo total de fricción**, la orden se descarta automáticamente con estado `REJECTED_BY_FRICTION` y queda registrada en la auditoría sin arriesgar capital.

### 4. Detección de Regímenes HMM 3-Estados & Kolmogorov-Smirnov Drift (`regime_detector.py`)
El LSTM Autoencoder unidimensional solo detecta si una vela es atípica, pero no sabe si el mercado está en tendencia o en consolidación caótica. `InstitutionalRegimeDetector` formaliza un modelo de Markov Oculto Gaussiano (**Gaussian HMM**) de 3 estados:

```
                   ┌──────────────────────────────────────────┐
                   │    REGÍMENES DE MERCADO INSTITUCIONALES  │
                   └────────────────────┬─────────────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
    ┌─────────────────┐        ┌─────────────────┐        ┌───────────────────────┐
    │ Estado 0:       │        │ Estado 1:       │        │ Estado 2:             │
    │ TENDENCIA ALTA  │        │ TENDENCIA BAJA  │        │ RANGO TURBULENTO/CHOP │
    │ (Bull Trend)    │        │ (Bear Trend)    │        │ (High Vol, No Trend)  │
    ├─────────────────┤        ├─────────────────┤        ├───────────────────────┤
    │ • Kelly: 1.0x   │        │ • Kelly: 1.0x   │        │ • Kelly: 0.25x / CASH │
    │ • Foco: Compras │        │ • Foco: Ventas  │        │ • Bloqueo de breakout │
    │ • Barreras TP+  │        │ • Barreras TP-  │        │ • Protege de serrucho │
    └─────────────────┘        └─────────────────┘        └───────────────────────┘
```

- **Monitor de Volatilidad Kolmogorov-Smirnov (KS-Test):**
  - Evalúa la función de distribución acumulada (CDF) de los retornos de las últimas 50 velas contra la distribución histórica in-sample (`scipy.stats.ks_2samp`).
  - Si el p-valor $< 0.01$, diagnostica formalmente **Regime Drift**: reduce preventivamente el riesgo de Kelly al **50% (Safety Factor = 0.50x)** y emite una alerta a Telegram para programar el reajuste de hiperparámetros.

### 5. Agente GenAI Macro-Hazard Pre-News y Reportes Forenses (`src/macro/`)
- **`macro_rag_agent.py`:** Protege el portafolio contra shocks exógenos de política monetaria (FOMC, CPI, Non-Farm Payrolls, Decisiones de Tasas de Interés Centrales):
  - **Ventana Pre-News (Congelamiento):** Congela la apertura de nuevas posiciones 30 minutos antes del anuncio programado (`FREEZE_PRE_NEWS`).
  - **Ventana Post-News (Cooldown):** Mantiene 15 minutos de enfriamiento posterior mientras se normaliza la liquidez del libro de órdenes.
  - **Bitácora Forense Post-Trade (RCA):** Si ocurre una salida por Stop Loss o drawdown inesperado, cruza el timestamp exacto con el calendario macroeconómico e imprime un informe estructurado de análisis de causa raíz para comités de inversión y Darwinex.
- **`economic_calendar.py`:** Proveedor de eventos macroeconómicos programados con persistencia local en caché JSON.

### 6. Bóveda Transaccional ACID In-Process (`src/database/trade_vault.py`)
Reemplaza los archivos planos `.json` y `.csv` de la v1.0 por una base de datos embebida **SQLite en modo Write-Ahead Logging (WAL)**:
- **Cero latencia de red:** Base de datos embebida in-process sin sobrecarga de sockets TCP.
- **Tolerancia a cortes de energía:** Transacciones ACID garantizadas ante reinicios abruptos de servidores VPS o instancias AWS EC2.
- **Auditabilidad Total:** Cada orden, probabilidad del ONNX, spread al momento del llenado, slippage, error MSE del autoencoder y ratio hurdle queda indexado en la tabla `execution_audit_log`.

---

## 📊 Matriz Comparativa de Evolución Arquitectónica (v1.0 vs. v2.0)

| Dimensión de Arquitectura | Quant Bot v1.0 (López de Prado Base) | Quant Bot v2.0 (AlphaEdge Sentinel) | Impacto Cuantitativo Medible |
| :--- | :--- | :--- | :--- |
| **Latencia Tick-to-Order** | 45 ms – 120 ms (Grafo Keras en CPU) | **< 1.8 ms (ONNX Runtime C++ BLAS)** | $\mathbf{\approx 25\times}$ más rápido; elimina slippage evitable |
| **Concurrencia de Procesos** | Monolítico (Reentrenamiento bloquea ejecución) | **Doble Motor Desacoplado (Gateway vs. Daemon)** | Ejecución 100% libre de retrasos en aperturas de sesión |
| **Huella de Memoria VPS** | ~1,200 MB – 1,600 MB (TensorFlow/Keras) | **< 130 MB RAM (ONNX + MT5 + NumPy)** | Despliegue seguro en VPS económicos; cero cuelgues OOM |
| **Sensibilidad a Fricción** | Teórica (Barreras puras de precio Mid/Close) | **Filtro de Utilidad Neta ($E[U] \ge 2.5 \times \text{Cost}$)** | Filtra trades marginales devorados por spread y swap |
| **Clasificación de Régimen** | Autoencoder P90/P99 (Detección de anomalías) | **3-State HMM + Rolling KS-Test + Autoencoder** | Reduce drawdown en rangos laterales (*choppy/whipsaw*) |
| **Gestión de Eventos Macro** | Ciego a anuncios fundamentales programados | **Macro Hazard Guard (Pre-News Freeze & RCA)** | Blindaje contra cisnes negros de política monetaria |
| **Persistencia de Estados** | Archivos planos `.json`, `.csv`, `.npy` | **SQLite WAL In-Process (`TradeVault` ACID)** | Cero corrupción de archivos por reinicios de VPS |
| **Gobernanza de Modelos** | Serialización `.pkl` manual | **Exportación ONNX con Checksum SHA-256** | Integridad criptográfica inmutable en producción |

---

## 🔌 Esquemas Transaccionales y Contratos de Auditoría

### 1. Esquema de la Tabla de Auditoría (`TradeVault`)
```sql
CREATE TABLE IF NOT EXISTS execution_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    signal_direction TEXT NOT NULL,         -- BUY, SELL, CASH
    model_name TEXT NOT NULL,
    prediction_prob REAL NOT NULL,
    hmm_regime_state INTEGER NOT NULL,      -- 0: Bull, 1: Bear, 2: Choppy
    autoencoder_mse REAL NOT NULL,
    expected_utility REAL NOT NULL,
    estimated_friction REAL NOT NULL,
    spread_points REAL NOT NULL,
    slippage_points REAL NOT NULL,
    swap_points REAL NOT NULL,
    hurdle_ratio REAL NOT NULL,
    kelly_multiplier REAL NOT NULL,
    lot_size REAL NOT NULL,
    entry_price REAL NOT NULL,
    tp_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    status TEXT NOT NULL,                    -- FILLED, REJECTED_BY_FRICTION, REJECTED_BY_MACRO, CASH
    reason TEXT,
    raw_metadata TEXT                        -- Metadatos JSON (VIX, DXY, tick timestamp)
);
```

### 2. Registro Transaccional de Decisión en Producción (`execution_audit_log`)
```json
{
  "timestamp_utc": "2026-10-15T14:30:01.218Z",
  "symbol": "EURUSD",
  "timeframe": "D1",
  "signal_direction": "BUY",
  "model_name": "XGBOOST_PRECIO_PURO",
  "prediction_prob": 0.7245,
  "hmm_regime_state": 0,
  "autoencoder_mse": 0.00084,
  "expected_utility": 18.25,
  "estimated_friction": 4.10,
  "spread_points": 1.20,
  "slippage_points": 0.65,
  "swap_points": 0.25,
  "hurdle_ratio": 4.45,
  "kelly_multiplier": 1.0,
  "lot_size": 0.24,
  "entry_price": 1.08550,
  "tp_price": 1.09450,
  "sl_price": 1.08100,
  "status": "FILLED",
  "reason": "APROBADO: Utilidad Neta Esperada 18.25 pts (Hurdle ratio: 4.45x >= 2.50x)",
  "latency_ms": 11.42
}
```

---

## 🏆 Separación de Entornos: Training Offline vs. Serving Ultra-Lean (< 130 MB)

Siguiendo las mejores prácticas de ingeniería de software y MLOps bancario, separamos las dependencias analíticas de las de ejecución en vivo:

| Ámbito | Componentes & Modelos | Entorno de Ejecución & Justificación MLOps |
| :--- | :--- | :--- |
| **Investigación & Torneo Offline** | `main_training.py`<br>`bilstm_model.py`<br>`arima_lstm.py`<br>`portfolio_backtester.py` | **`requirements/research.txt` (TensorFlow / Keras / Optuna / Scikit-learn):**<br>Entorno completo para entrenamiento de redes neuronales, optimización Bayesiana con poda de hiperparámetros (TPE Sampler) y validación combinatoria CPCV. |
| **Inferencia en Producción (Serving)** | `main_bot_v2.py`<br>`execution_gateway.py`<br>`onnx_inference_engine.py`<br>`cost_sensitive_gatekeeper.py` | **`requirements/execution.txt` (ONNX Runtime / MT5 / NumPy / Requests):**<br>Entorno ultra-lean para el servidor VPS en vivo. Purga completamente TensorFlow y Keras, logrando un consumo de RAM **menor a 130 MB** con tiempo de inicio inferior a 0.5 segundos y cero riesgo de cuelgues OOM. |

---

## 🧠 Marco Metodológico de Marcos López de Prado (Core Cuantitativo)

El bot preserva e integra el marco matemático riguroso de *Advances in Financial Machine Learning*:

1. **Fractional Differencing (FFD):** Encuentra analíticamente el orden fraccionario mínimo $d^*$ mediante el test de Dickey-Fuller Aumentado (ADF), alcanzando estacionariedad estricta sin borrar la memoria de largo plazo ($\text{Corr} > 0.85$).
2. **Triple Barrier Method & Meta-Labeling:** Salidas dinámicas definidas por dos barreras horizontales fijadas por la volatilidad condicional diaria $\sigma_t$ (EGARCH) y una barrera vertical temporal (`Max Hold` de 10 días).
3. **Volatilidad Condicional EGARCH(1,1):** Modela la asimetría del apalancamiento financiero y el clustering de volatilidad para calibrar el ancho de las barreras en tiempo real.
4. **Hierarchical Risk Parity (HRP) con Shrinkage Bayesiano:** Asigna pesos de portafolio sin inversión de matrices de covarianza inestables mediante teoría de grafos, agrupamiento jerárquico (Tree Clustering) y quasi-diagonalización, ponderado por el ratio Sharpe rodante de 100 días ($\lambda = 0.85$).
5. **Auditoría CPCV & Probability of Backtest Overfitting (PBO):** Combinatorial Purged Cross-Validation dividiendo la historia en combinaciones $\binom{6}{2} = 15$ caminos cruzados para certificar que el Sharpe out-of-sample no es fruto de la minería de datos ($PBO < 5\%$).
6. **Walk-Forward Rolling Monte Carlo MDD (NumPy 2D Vectorizado):** 1,000 permutaciones resampleadas calculadas cada 30 trades en vivo para dimensionar el riesgo de Kelly sin ningún sesgo de anticipación (*Lookahead Bias*).

---

## 📦 Estructura del Repositorio

```text
quant-trading-bot/
├── data/
│   ├── raw/                              # Velas históricas crudas MT5 (CSV)
│   └── processed/                        # Datasets con FFD, EGARCH y exógenas
├── requirements/
│   ├── execution.txt                     # [v2.0] Ultra-lean para VPS (< 130 MB RAM)
│   └── research.txt                      # [v2.0] Suite de investigación y entrenamiento DL
├── src/
│   ├── database/                         # [v2.0] Capa de Persistencia Transaccional ACID
│   │   ├── __init__.py
│   │   └── trade_vault.py                # SQLite WAL In-Process TradeVault
│   ├── execution/                        # Motores de Ejecución y Gestión de Riesgo
│   │   ├── analytics_daemon.py           # [v2.0] Demonio Asíncrono MLOps (Shadow Journal, HRP)
│   │   ├── cost_sensitive_gatekeeper.py  # [v2.0] Filtro de Fricción y Hurdle Rate (2.5x)
│   │   ├── execution_engine.py           # Conexión directa a deals MT5
│   │   ├── execution_gateway.py          # [v2.0] Gateway de ultra-baja latencia (< 15 ms)
│   │   ├── main_bot_v2.py                # [v2.0] Orquestador Institucional (Dual/Gateway/Daemon)
│   │   ├── main_bot.py                   # Bot legacy v1.0 (retrocompatibilidad)
│   │   ├── onnx_inference_engine.py      # [v2.0] Motor de Inferencia C++ BLAS (< 1.8 ms)
│   │   ├── risk_manager.py               # Position Sizing Kelly y Monitor Híbrido
│   │   └── telegram_notifier.py          # Notificaciones push asíncronas
│   ├── macro/                            # [v2.0] Inteligencia Macroeconómica Pre-News
│   │   ├── __init__.py
│   │   ├── economic_calendar.py          # Calendario de eventos fundamentales y caché
│   │   └── macro_rag_agent.py            # Pre-News Hazard Guard (-30m/+15m) y RCA Forense
│   ├── models/                           # Modelos Predictivos y Explicabilidad
│   │   ├── anomaly_detector.py           # LSTM Autoencoder y Gaussian HMM
│   │   ├── arima_lstm.py                 # Híbrido ARIMA-LSTM
│   │   ├── bilstm_model.py               # BiLSTM Bidireccional
│   │   ├── export_champions_to_onnx.py   # [v2.0] Script de exportación masiva a ONNX
│   │   ├── lstm_model.py                 # LSTM Recurrente
│   │   ├── lstm_rf.py                    # Híbrido LSTM-RandomForest
│   │   ├── onnx_exporter.py              # [v2.0] Constructor de grafos y Checksums SHA-256
│   │   ├── random_forest.py              # Random Forest Classifier
│   │   ├── regime_detector.py            # [v2.0] HMM 3-Estados y Test Kolmogorov-Smirnov
│   │   └── xgb_model.py                  # XGBoost Classifier
│   ├── preprocessing/                    # Ingeniería de Características Financieras
│   │   ├── asset_screener.py             # Screening Fase 0 (Hurst > 0.55, Correlación < 0.40)
│   │   ├── auditor.py                    # Auditoría de integridad de datasets
│   │   ├── chilean_macro.py              # Exógenas macro chilenas / globales
│   │   ├── meta_labeling.py              # Etiquetado con Triple Barrera
│   │   ├── stationarity.py               # Diferenciación Fraccionada (FFD)
│   │   ├── technical_features.py         # RSI, MACD, ATR, Bollinger
│   │   └── volatility.py                 # Calibración condicional EGARCH
│   ├── evaluation/                       # Torneo de Modelos y Asignación de Portafolio
│   │   ├── alpha_backtester.py           # Backtester vectorizado de Alpha
│   │   ├── cpcv_auditor.py               # Auditoría CPCV y métrica PBO
│   │   ├── hrp_optimizer.py              # Hierarchical Risk Parity
│   │   ├── live_evaluator.py             # Auditoría de rendimiento en vivo
│   │   └── portfolio_backtester.py       # Simulador financiero en USD y selección de campeones
│   ├── data_extractor.py                 # Extractor histórico MT5
│   ├── main_preprocessing.py             # Pipeline maestro de preprocesamiento incremental
│   ├── main_training.py                  # Pipeline de optimización Bayesiana (Optuna)
│   └── mt5_connector.py                  # Conector nativo de IPC con MetaTrader 5
├── tests/                                # Suite Completa de 63 Pruebas Unitarias Pytest
│   ├── conftest.py                       # Fixtures sintéticos (bullrun, crash, flat, clean)
│   ├── test_bayesian_opt.py              # Validación de Optuna TPE sampler
│   ├── test_cost_sensitive_gatekeeper.py # [v2.0] Hurdle rate, slippage y spreads
│   ├── test_cpcv_auditor.py              # Combinaciones CPCV y Sharpe medio
│   ├── test_decoupled_architecture.py    # [v2.0] Integración Gateway vs. Analytics Daemon
│   ├── test_egarch.py                    # Volatilidad positiva y cap de sanidad
│   ├── test_ffd.py                       # Estacionariedad y preservación de memoria
│   ├── test_macro_rag_agent.py           # [v2.0] Ventanas pre/post news y RCA
│   ├── test_multi_timeframe.py           # Sincronización horaria D1/H4
│   ├── test_onnx_inference.py            # [v2.0] Inferencia C++, shapes y SHA-256
│   ├── test_regime_detector.py           # [v2.0] HMM 3 estados y test KS de deriva
│   ├── test_risk_manager.py              # Position sizing, Kelly y barreras
│   ├── test_technical_features.py        # Límites de osciladores y consistencia
│   ├── test_trade_vault.py               # [v2.0] Persistencia ACID y modo WAL SQLite
│   └── test_triple_barrier.py            # Lógica de etiquetado de barreras
├── results/                              # Artefactos de producción (Ignorado por Git)
│   ├── campeon_*.json                    # Configuraciones de modelos campeones
│   ├── saved_models/                     # Modelos (.pkl, .keras, .onnx) y onnx_manifest.json
│   ├── trading_vault.db                  # [v2.0] Bóveda transaccional SQLite WAL
│   └── hrp_weights.json                  # Matriz de pesos de portafolio HRP vigente
├── requirements.txt                      # Dependencias base
└── README.md                             # Documentación Técnica Institucional
```

---

## 🛠️ Instalación y Verificación Rápida

### 1. Clonar e Instalar Entorno Virtual
```bash
git clone https://github.com/ronaldreighsrsc/quant-trading-bot.git
cd quant-trading-bot

# Crear y activar entorno virtual Python 3.12
python -m venv venv
venv\Scripts\activate       # En Windows
# source venv/bin/activate  # En Linux/Mac

# Instalar dependencias completas
pip install -r requirements.txt
```

### 2. Ejecutar la Suite de Pruebas Automatizadas
```bash
pytest tests/ -v
```
**Resultado:** **63 pruebas unitarias e integradas aprobadas al 100% (63 passed in ~95s)**:
- `test_trade_vault.py` (4 tests): Inicialización WAL, concurrencia de hilos y auditoría.
- `test_cost_sensitive_gatekeeper.py` (4 tests): Hurdle rate $\ge 2.5\times$, slippage estocástico y spreads.
- `test_regime_detector.py` (5 tests): Mapeo semántico HMM 3-estados y test Kolmogorov-Smirnov ($p < 0.01$).
- `test_macro_rag_agent.py` (6 tests): Pre-News freeze, cooldown post-news y reporte forense RCA.
- `test_onnx_inference.py` (5 tests): Inferencia continua `float32`, verificación SHA-256 y latencia.
- `test_decoupled_architecture.py` (3 tests): Abort O(1) en cuarentena, macro freeze y latido de vida.
- `test_risk_manager.py`, `test_ffd.py`, `test_egarch.py`, etc. (36 tests): Métricas base de López de Prado.

### 3. Puesta en Marcha del Bot en Producción (v2.0)
```bash
# Modo Dual Unificado (Gateway en tiempo real < 15ms + Daemon Analítico MLOps)
python src/execution/main_bot_v2.py --mode dual --interval 5.0

# Modo Solo Gateway (Para VPS de 1 GB RAM ultra-reducido sin TensorFlow)
python src/execution/main_bot_v2.py --mode gateway --interval 5.0

# Modo Solo Demonio Analítico (Worker secundario de rebalanceo y Shadow Journal)
python src/execution/main_bot_v2.py --mode daemon
```

### 4. Zero-Touch MLOps: Compilación Automática a ONNX & Manifiesto Criptográfico

> [!TIP]
> **Recomendación Operacional Táctica para Reentrenamiento & Producción (Zero-Touch MLOps):**  
> Para evitar desalineaciones en producción y garantizar cero errores humanos, **unificamos ambas responsabilidades en el pipeline principal**:  
> Siempre que ejecutes el reentrenamiento periódico de modelos con `python src/main_training.py` o el torneo de asignación de portafolio con `python src/evaluation/portfolio_backtester.py`, el sistema **compila y exporta automáticamente** los modelos campeones a formato `.onnx` en `results/saved_models/` y refresca sus firmas criptográficas en `onnx_manifest.json`.
>
> De este modo, el flujo operativo heredado (*legacy*) no se altera y `execution_gateway.py` siempre arranca con las firmas criptográficas al día de forma 100% desatendida.

Si deseas forzar una re-exportación o validación manual aislada sin reentrenar (p. ej. en un job de CI/CD o tras migrar un modelo entre servidores):
```bash
python src/models/export_champions_to_onnx.py
```
*Actualiza los archivos `.onnx` en `results/saved_models/` y calcula los checksums SHA-256 en `onnx_manifest.json`.*

### 5. Reproducción del Pipeline Completo de Investigación (ETL, Optuna & Portafolio)
```bash
# 1. Extracción y screening nativo MT5 (Hurst > 0.55)
python src/data_extractor.py

# 2. Preprocesamiento incremental (FFD + EGARCH + Triple Barrera)
python src/main_preprocessing.py

# 3. Retuning Bayesiano con Optuna (Poda de hiperparámetros)
#    -> Guarda modelos campeones y compila automáticamente a ONNX con firmas SHA-256
python src/main_training.py

# 4. Torneo de Portafolio Financiero y Asignación HRP
#    -> Exporta hrp_weights.json y sincroniza los grafos ONNX para Producción
python src/evaluation/portfolio_backtester.py
```

### 6. Empaquetado Inteligente para AWS / VPS (`export_to_aws.py`)
Genera un bundle optimizado `bot_production.zip` que incluye exclusivamente los artefactos requeridos en vivo:
```bash
python export_to_aws.py
```
*Empaqueta los modelos campeones (`.onnx`, `.pkl`, `.keras`), monitores MLOps (`results/campeon_*_autoencoder/`, `results/campeon_*_hmm.pkl`), el manifiesto criptográfico `onnx_manifest.json`, los pesos de portafolio `hrp_weights.json`, el código fuente `src/`, el script de servicio `start_bot.bat` y los requerimientos de ejecución.*

---

## ☁️ Despliegue y Automatización 24/7 en Servidor AWS / EC2 / Windows VPS

Para que el bot opere de forma ininterrumpida y sobreviva a reinicios automáticos de instancias cloud o parches de seguridad de Windows Server, **es imperativo considerar la restricción gráfica de MetaTrader 5**:

> [!WARNING]
> **Restricción de Entorno GUI de MetaTrader 5:**  
> MetaTrader 5 requiere inicializar subsistemas gráficos de Windows para interactuar con la API nativa de IPC. Por ello, **NO** debe lanzarse como un servicio silencioso de Windows ("Session 0 Isolation"), ya que crashearía de forma inmediata al no encontrar un contexto gráfico activo.

Para configurar la ejecución automática e indestructible en una instancia **AWS EC2 (Windows Server)** o VPS dedicado:

### Paso 1: Configurar Auto-Login en Windows Server
1. Abre el Símbolo del Sistema (CMD) como **Administrador** y destraba la directiva de contraseñas de Windows:
   ```cmd
   reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device" /v DevicePasswordLessBuildVersion /t REG_DWORD /d 0 /f
   ```
2. Presiona `Win + R`, escribe `netplwiz` y pulsa Enter.
3. Desmarca la casilla *"Los usuarios deben escribir su nombre y contraseña para usar el equipo"*.
4. Haz clic en **Aplicar**, introduce la contraseña de `Administrator` dos veces y acepta.  
   *(Con esto, el servidor iniciará sesión automáticamente en el escritorio al encender o reiniciar).*

### Paso 2: Programar la Tarea con Privilegios Elevados (`schtasks`)
Crea la tarea programada que ejecutará el script de arranque justo cuando el escritorio cargue:
```cmd
schtasks /create /tn "QuantBot_Trading" /tr "C:\Users\Administrator\Desktop\quant-trading-bot\start_bot.bat" /sc onlogon /ru "Administrator" /rl highest /f
```

### Script de Arranque Resiliente (`start_bot.bat`)
El script incorpora una pausa inicial de 60 segundos para permitir que MetaTrader 5 y los servicios de red sincronicen su conexión con el broker antes de invocar el orquestador institucional en modo dual:
```bat
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
```

---

## 📅 Calendario de Mantenimiento MLOps Institucional

Para asegurar una operación libre de drift y sin confusión de roles operativos, sigue esta cadencia recomendada:

| Fase MLOps | Frecuencia Recomendada | Script a Ejecutar | Parámetro / Detalle Clave | Qué hace / Qué genera |
| :--- | :--- | :--- | :--- | :--- |
| **1. Refresco de Datos** | Mensual | `python src/data_extractor.py`<br>`python src/main_preprocessing.py` | N/A | Descarga velas MT5 recientes y actualiza matrices con FFD y EGARCH en `data/processed/*.csv`. |
| **2. Reentrenamiento de Modelos** | Semestral / Anual | `python src/main_training.py` | N/A | Optimización Bayesiana Optuna. **Auto-compila a ONNX con firmas SHA-256 en `onnx_manifest.json` (Zero-Touch MLOps).** |
| **3. Monitores MLOps & Torneo** | Semestral / Tras Paso 2 | `python src/evaluation/portfolio_backtester.py` | `fast_mode=False` (~1 h) | Entrena detectores HMM, Autoencoders de anomalías y sincroniza los grafos `.onnx` para producción. |
| **4. Auditoría de Robustez PBO** | Trimestral / Tras Paso 2 | `python src/evaluation/cpcv_auditor.py` | N/A | Evalúa $\binom{6}{2}=15$ caminos combinatorios CPCV certificando $PBO < 5\%$. |
| **5. Rebalanceo de Pesos HRP** | Mensual (ej. día 1) | `python src/evaluation/portfolio_backtester.py` | `fast_mode=True` (~2 min) | Recalcula la matriz HRP con Shrinkage Sharpe rodante y actualiza `results/hrp_weights.json`. |
| **6. Empaquetado Producción** | Tras Paso 2 o 5 | `python export_to_aws.py` | N/A | Genera el bundle `bot_production.zip` listo para desplegar en la instancia AWS / VPS. |
| **7. Operativa en Vivo 24/7** | Continua (Mercado Abierto) | `python src/execution/main_bot_v2.py --mode dual` | `--interval 5.0` | Ejecución en ultra-baja latencia (< 15 ms), Shadow Journal asíncrono y auditoría SQLite WAL. |

---

## 📓 Auditoría de Producción en Vivo & Dashboards

### 1. Evaluador de Rendimiento en Vivo (`src/evaluation/live_evaluator.py`)
Permite auditar el desempeño financiero real del bot en caliente sin depender exclusivamente de los reportes del broker:
```bash
python src/evaluation/live_evaluator.py
```
Genera un informe interactivo HTML en `results/live_production_report.html` con:
- Equidad y Balance en tiempo real.
- Retorno sobre la Inversión (ROI) y Sharpe Ratio en vivo.
- Máximo Drawdown real vs. Drawdown proyectado por Monte Carlo.
- Historial detallado de operaciones cerradas y tasa de acierto (Win Rate).

### 2. Consultas Forenses en la Bóveda Transaccional (`results/trading_vault.db`)
Gracias al modo SQLite Write-Ahead Logging (WAL), puedes consultar la base de datos en caliente sin bloquear la ejecución del bot:
```bash
sqlite3 results/trading_vault.db "SELECT timestamp_utc, symbol, signal_direction, expected_utility, estimated_friction, hurdle_ratio, status FROM execution_audit_log ORDER BY id DESC LIMIT 10;"
```

### 3. Dashboard MLflow (Experiment Tracking & Model Registry)
Visualiza el historial completo de entrenamientos, torneos y métricas CPCV:
```bash
mlflow ui
```
Disponible en `http://127.0.0.1:5000` con tracking de parámetros de Optuna, curvas de aprendizaje, matrices de confusión y artefactos exportados.

