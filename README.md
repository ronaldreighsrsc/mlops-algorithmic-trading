# Quantitative Trading Research & Execution Pipeline (Python / MT5)

Laboratorio de ingeniería cuantitativa y MLOps aplicado a series temporales financieras, basado en la metodología de *Advances in Financial Machine Learning* (Marcos López de Prado) y desacoplado en entrenamiento offline e inferencia ligera con **ONNX Runtime** y **SQLite WAL**.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Pytest](https://img.shields.io/badge/Tests-102%2F102_Passing-success?style=flat-square&logo=pytest)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Parity_<1e--5-005ced?style=flat-square&logo=onnx)
![Persistence](https://img.shields.io/badge/SQLite_WAL-ACID_In--Process-003B57?style=flat-square&logo=sqlite)
![Validation](https://img.shields.io/badge/Validation-CSCV_PBO_%26_DSR-orange?style=flat-square)

---

## 1. Alcance, Enfoque y Limitaciones del Ecosistema

Este repositorio nació como un pipeline de trading algorítmico sobre **MetaTrader 5 (MT5)** y evolucionó hacia un sistema de investigación y validación reproducible enfocado en resolver tres problemas reales de ingeniería de datos y ML financiero:

1. **Prevenir *data leakage* temporal y *backtest overfitting*:** validación purgada por intervalos `[t0, t1]`, pesos por unicidad de muestra, poda de features *out-of-sample* (MDA) y auditoría estadística **CSCV PBO** + **Deflated Sharpe Ratio (DSR)**.
2. **Separar generación de hipótesis de filtrado estadístico (Meta-Labeling real en 2 etapas):** evitar obligar a modelos complejos a predecir dirección diaria en días sin información.
3. **Aislar dependencias entre entrenamiento e inferencia:** evitar cargar ~1.4 GB de TensorFlow/Keras en un servidor de ejecución ligero mediante exportación real a **ONNX** (`< 130 MB` RAM) y sincronización no bloqueante vía **SQLite WAL**.

> **Nota de Contexto Institucional:** El conector de mercado utiliza **MetaTrader 5** (entorno *retail* / *prop* sobre CFDs y FX). No implementa protocolos institucionales de renta fija o acciones locales como **FIX**, feeds **Bloomberg** ni reglas regulatorias de un OMS corporativo bajo normativa CMF. El foco técnico del proyecto es la **ingeniería de datos temporales, validación estadística sin sesgos y MLOps reproducible**.

---

## 2. Resultados Empíricos Reales & Decisiones de Descarte (v3.0 → v3.1)

Al auditar la versión previa del proyecto (v1.0/v2.0), se detectaron y corrigieron **sobreparametrización** (`BiLSTM` de ~300K–1.2M parámetros sobre ~3,000 velas diarias), **Pseudo Meta-Labeling** (etiquetado Long-only sin modelo primario ni filtro de eventos), **fuga temporal de `max_hold=10` días** en el reentrenamiento *walk-forward*, ***threshold snooping*** directo sobre el test set OOS, y un exportador ONNX prototipo que no reflejaba pesos reales.

Posteriormente, en **v3.1** (`docs/SPEC_QUANT_V3_1_ALPHA_SIGNALS.md`) se incorporó un extractor causal macro/volatilidad (`MacroRatesProvider`: series FRED + `yfinance` + `XAGUSD` con `T_macro < T_asset`, cero `.bfill()`) y dos estrategias primarias económicas (`MacroCarryMomentumPrimary`, `VixRegimeEquityPrimary`) evaluadas bajo un **torneo pre-registrado cerrado ($N = 6$: 3 primarias × 2 modelos tabulares)** sobre el mismo holdout OOS `70/30` y 5 compuertas duras (`OOS LogLoss < 0.67`, `ECE < 0.05`, `DSR >= 0.95`, `CSCV PBO < 20%`, `OOS Net Sharpe > Benchmark Sharpe`).

Estos son los **resultados reales reproducibles *out-of-sample*** ejecutando `python src/main_training_v3.py --compare`:

### Comparativa OOS: Baseline `v3.0 (d1_ohlcv)` vs. `v3.1 (d1_macro, N=6)`

| Activo | Versión / Modo | Estrategia + Modelo Ganador | Features MDA | OOS LogLoss | OOS ECE | OOS Sharpe (vs Bench) | DSR Empírico | **CSCV PBO** | Compuerta (5/5) | ONNX Diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EURUSD`** | `v3.0 (d1_ohlcv)` | `RANDOM_FOREST` (Calibrado) | 7 | `0.7057` | `0.0862` | — | `0.005` | **`82.86%`** | N/A | `1.07e-07` |
| **`EURUSD`** | `v3.1 (d1_macro)` | `DonchianBreakout` + `RF` | 18 | `0.6992` | `0.0771` | `-0.29` (`0.17`) | `0.032` | **`22.86%`** | **Rechazado (0/5)** | `1.15e-07` |
| **`Oro`** | `v3.0 (d1_ohlcv)` | `RANDOM_FOREST` (Calibrado) | 15 | `0.7110` | `0.1100` | — | `0.998` | **`75.71%`** | N/A | `1.91e-07` |
| **`Oro`** | `v3.1 (d1_macro)` | `MacroCarryMomentum` + `RF` | 19 | `0.6939` | `0.0773` | **`1.26`** (`1.21`) | `0.630` | **`0.00%`** | **Rechazado (2/5)** | `2.01e-07` |
| **`SP500`** | `v3.0 (d1_ohlcv)` | `RANDOM_FOREST` (Calibrado) | 8 | `0.6906` | `0.0687` | — | `0.879` | **`52.86%`** | N/A | `8.43e-08` |
| **`SP500`** | `v3.1 (d1_macro)` | `VixRegimeEquity` + `RF` | 9 | `0.6870` | `0.0582` | **`1.18`** (`0.86`) | `0.865` | **`30.00%`** | **Rechazado (1/5)** | `1.42e-07` |

### Qué muestran estos números (y qué se descartó)
- **Por qué se descartaron las redes `BiLSTM` / `LSTM` en `D1`:** Con ~2,500–4,000 barras diarias y baja relación señal/ruido, las redes recurrentes profundas memorizaban ruido y ganaban torneos por selección múltiple sin filtro DSR. En el torneo v3.1 ($N=6$), se fijaron únicamente modelos tabulares regularizados (`Logistic L2` y `RandomForest` con `max_depth=5`, `max_samples = avg_uniqueness`).
- **Impacto causal de señales macro (`v3.0` → `v3.1`):** Incorporar diferenciales de tasas (`rate_diff_us2y_ecb`, `rate_diff_fed_ecb`), tasa real TIPS (`us_real_yield_10y`), ratio `XAU/XAG`, estructura VIX (`VIX/VIX3M`) y estacionalidad de fin de mes redujo el **CSCV PBO** en los tres activos (`82.86% → 22.86%` en `EURUSD`, `75.71% → 0.00%` en `Oro`, `52.86% → 30.00%` en `SP500`), mejoró `OOS LogLoss` y `OOS ECE` en los tres, y permitió que `MacroCarryMomentum` (`Oro`) y `VixRegimeEquity` (`SP500`) superaran en Sharpe neto a sus *benchmarks* pasivos (`1.26 vs 1.21` y `1.18 vs 0.86`).
- **Disciplina de compuerta (`gate_approved = False`):** Pese a la mejora relativa frente a `d1_ohlcv`, **ninguno de los tres activos en `D1` aprobó simultáneamente las 5 compuertas duras** (principalmente `OOS LogLoss < 0.67` y `ECE < 0.05`). El auditor **rechaza desplegar estas señales diarias con capital real** en lugar de relajar umbrales *post-hoc*; el camino natural habilitado en el código (`--mode intraday_tib`) es migrar hacia **Tick Imbalance Bars + microestructura VPIN** una vez extraídos ticks históricos reales desde MT5.

---

## 3. Arquitectura del Sistema (4 Capas Cuantitativas + Ejecución Desacoplada)

### 3.1 Flujo de Señal y Validación (v3.0 / v3.1 — `src/main_training_v3.py`)

```text
[Datos OHLCV + Macro FRED/yfinance + Ticks MT5]
   │
   ├──► Capa 0: Muestreo por Información & Exógenas Causales
   │      • Macro Causal (macro_rates.py): FRED + VIX/VIX3M + HYG/IEF + XAU/XAG (T_macro < T_asset)
   │      • D1: Filtro CUSUM Simétrico (umbral h_t = κ · σ_{t-1, EGARCH})
   │      • Intradía (intraday_tib): Filtro sesión 07–17 UTC + Tick Imbalance Bars + VPIN (information_bars.py)
   │      • Screening: Hurst sobre retornos estacionarios + Variance Ratio Lo-MacKinlay
   │
   ├──► Capa 1: Modelo Primario Causal (src/strategies/primary_strategies.py)
   │      • Técnicas D1: TrendPullback, DonchianBreakout, VolatilityMeanReversion
   │      • Económicas v3.1: MacroCarryMomentum (tasas/TIPS), VixRegimeEquity (VIX3M/crédito)
   │      • Emite únicamente dirección candidata: side ∈ {-1, 0, +1}
   │
   ├──► Capa 2: Meta-Modelo Tabular Calibrado (src/models/meta_trainer_v3.py)
   │      • MetaBarrierLabeler orientado por `side` con intervalos exactos [t0, t1]
   │      • Pesos por unicidad de muestra (concurrency + time-decay, sample_weights.py)
   │      • Features tabulares causales v3.1 (tabular_features.py, sin .bfill())
   │      • Poda OOS por MDA Purgado (feature_importance.py)
   │      • Torneo Pre-Registrado N=6 (3 primarias x {LogL2, RF}) + 5 compuertas (LogLoss, ECE, DSR, PBO, Sharpe)
   │      • Calibración PurgedPlattCalibrator sobre OOF purgado + auditoría CSCV PBO (T x N)
   │
   └──► Capa 3: Serving ONNX & Control de Riesgo (src/execution/)
          • Exportación Pipeline real (Imputer + Scaler + Modelo + Platt) -> .onnx + SHA-256
          • Compuerta obligatoria de paridad numérica < 1e-5 contra modelo Python
          • Política Fail-Closed ante corrupción o hash inválido (sin fallback silencioso)
```

### 3.2 Ejecución Desacoplada en Producción (`main_bot_v2.py`)

El runtime separa el loop de trading del trabajo analítico pesado sincronizándolos mediante **SQLite en modo WAL (`results/trading_vault.db`)**:

- **`execution_gateway.py` (Proceso 1 — Loop de Trading):** consulta el estado de cuarentena en $O(1)$ desde SQLite WAL, valida ventanas de calendario macro (`MacroHazardGuard`, bloqueo `-30m / +15m` alrededor de FOMC/CPI/NFP desde caché JSON), evalúa el modelo ONNX (`< 130 MB` de RAM al no importar TensorFlow) aplicando `ffd_d_map` congelado, verifica utilidad neta esperada frente a costos estimados (`CostSensitiveGatekeeper`: $E[U] \ge 2.5 \times \text{fricción}$) y registra cada decisión (aprobada o rechazada) en `execution_audit_log`.
  - *Nota operacional sobre latencia:* en operativa de velas diarias (`D1`) con polling (`--interval 5.0`), inferir en `< 2 ms` vía ONNX no busca competir en HFT ni reducir slippage intrabar; el beneficio práctico en producción es **eliminar la huella de memoria de ~1.4 GB de TensorFlow/Keras** en servidores VPS económicos y mantener el loop libre de bloqueos.
- **`analytics_daemon.py` (Proceso 2 — Worker Asíncrono):** ejecuta fuera de la ruta crítica la evaluación del Autoencoder sobre telemetría de trades realizados `[net_ret, duration, win_flag]` (alertas P90/P99), actualización mensual de pesos **HRP** (*Hierarchical Risk Parity* con fallback a `1/N`) y publicación de estado en `TradeVault`.

### 3.3 Separación de Entornos

| Ámbito | Módulos | Dependencias & Justificación |
| :--- | :--- | :--- |
| **Investigación y Entrenamiento** | `main_training_v3.py`, `meta_trainer_v3.py`, `portfolio_backtester.py`, `cpcv_auditor.py` | `requirements/research.txt` (`scikit-learn`, `xgboost`, `lightgbm`, `skl2onnx`, `onnxmltools`, `optuna`, `statsmodels`, `arch`, `tensorflow`) |
| **Ejecución en Producción** | `main_bot_v2.py`, `execution_gateway.py`, `onnx_inference_engine.py`, `cost_sensitive_gatekeeper.py` | `requirements/execution.txt` (`MetaTrader5`, `onnxruntime`, `numpy`, `pandas`, `requests`, `python-dotenv`) — sin TensorFlow/Keras (`< 130 MB` RAM) |

---

## 4. Estructura del Repositorio

```text
quant-trading-bot/
├── docs/
│   ├── SPEC_QUANT_V3_INSTITUTIONAL_ALPHA.md  # Spec completo de auditoría y refactor v3.0
│   └── SPEC_QUANT_V3_1_ALPHA_SIGNALS.md      # Spec v3.1: señales macro, microestructura TIB y torneo N=6
├── requirements/
│   ├── execution.txt                     # Entorno ligero de inferencia (< 130 MB RAM)
│   └── research.txt                      # Entorno completo de investigación y exportación ONNX
├── src/
│   ├── database/
│   │   └── trade_vault.py                # Persistencia ACID SQLite WAL (execution_audit_log, system_state)
│   ├── strategies/
│   │   └── primary_strategies.py         # Capa 1: TrendPullback, Donchian, MeanReversion, MacroCarry, VixRegime
│   ├── preprocessing/
│   │   ├── asset_screener.py             # Hurst sobre retornos + Variance Ratio Lo-MacKinlay
│   │   ├── auditor.py                    # Auditoría de integridad y continuidad temporal
│   │   ├── chilean_macro.py              # Exógenas macro locales/globales
│   │   ├── cusum_filter.py               # Capa 0: Filtro CUSUM Simétrico calibrado por volatilidad
│   │   ├── information_bars.py           # Capa 0 intradía: sesión líquida, TIBs & microestructura VPIN
│   │   ├── macro_rates.py                # Proveedor causal FRED + yfinance + XAU/XAG (T_macro < T_asset)
│   │   ├── meta_labeling.py              # TripleBarrierLabeler + MetaBarrierLabeler ([t0, t1] por side)
│   │   ├── sample_weights.py             # Concurrencia, unicidad media y time-decay (AFML Cap. 4)
│   │   ├── stationarity.py               # FFD con d_map congelable (transform_with_fixed_d)
│   │   ├── tabular_features.py           # Features tabulares causales v3.0/v3.1 (sin .bfill())
│   │   ├── technical_features.py         # MACD, RSI, ATR
│   │   └── volatility.py                 # EGARCH(1,1) rodante con fallback a GARCH/std
│   ├── validation/
│   │   └── purged_cv.py                  # PurgedKFold([t0, t1]) + máscara causal Walk-Forward
│   ├── models/
│   │   ├── calibration.py                # PurgedPlattCalibrator + Brier/ECE
│   │   ├── export_champions_to_onnx.py   # Exportador CLI real (sin grafos sintéticos)
│   │   ├── meta_trainer_v3.py            # Entrenador Meta-Labeling v3.0 + exportación ONNX real
│   │   ├── onnx_exporter.py              # Conversión skl2onnx/onnxmltools + paridad < 1e-5 + SHA-256
│   │   ├── regime_detector.py            # HMM 3-estados + test Kolmogorov-Smirnov (KS-Drift)
│   │   ├── anomaly_detector.py           # Autoencoder (P90/P99) y HMM legacy
│   │   └── ...                           # Entrenadores legacy (xgb, rf, lstm, bilstm, arima_lstm, lstm_rf)
│   ├── evaluation/
│   │   ├── alpha_backtester.py           # Backtester causal (Open[t+1])
│   │   ├── cpcv_auditor.py               # CSCV PBO formal (matriz T x N), DSR empírico y CPCV
│   │   ├── feature_importance.py         # MDA Purgado Out-of-Sample por Log-Loss
│   │   ├── hrp_optimizer.py              # Hierarchical Risk Parity
│   │   ├── live_evaluator.py             # Reporte HTML de rendimiento en vivo
│   │   └── portfolio_backtester.py       # Simulador de portafolio y asignación HRP
│   ├── execution/
│   │   ├── analytics_daemon.py           # Worker asíncrono (Autoencoder 3-features, HRP)
│   │   ├── cost_sensitive_gatekeeper.py  # Filtro de utilidad neta vs spread/slippage/swap
│   │   ├── execution_engine.py           # Bridge directo a órdenes MT5
│   │   ├── execution_gateway.py          # Gateway Fail-Closed + flujo 4 capas v3.0
│   │   ├── main_bot_v2.py                # Orquestador (--mode dual | gateway | daemon)
│   │   ├── onnx_inference_engine.py      # Runtime ONNX con verificación SHA-256
│   │   ├── risk_manager.py               # Position sizing Kelly fraccional + HybridRiskMonitor
│   │   └── telegram_notifier.py          # Alertas asíncronas
│   ├── macro/
│   │   ├── economic_calendar.py          # Caché JSON de eventos macro
│   │   └── macro_rag_agent.py            # Hazard Guard (-30m/+15m) y bitácora RCA post-trade
│   ├── data_extractor.py                 # Extractor histórico MT5 (velas + ticks chunked) / yfinance
│   ├── main_preprocessing.py             # Pipeline incremental de preprocesamiento (--force)
│   ├── main_training_v3.py               # Pipeline v3.0 / v3.1 (--mode d1_ohlcv|d1_macro|intraday_tib, --compare)
│   └── main_training.py                  # Pipeline legacy multi-arquitectura
├── tests/                                # 102 pruebas unitarias y de integración (100% passing)
├── export_to_aws.py                      # Empaquetador bot_production.zip
├── pytest.ini                            # Configuración de pythonpath y testpaths
└── start_bot.bat                         # Script de arranque con espera GUI MT5
```

---

## 5. Instalación, Pruebas y Reproducción

### 5.1 Instalación y Suite de Pruebas
```bash
git clone https://github.com/ronaldreighsrsc/quant-trading-bot.git
cd quant-trading-bot

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
pytest tests/ -v
```
**Estado verificado:** **102 pruebas pasando (`102 passed`)**, cubriendo:
- Paridad numérica ONNX real `< 1e-5` (LR, RF, XGBoost, LightGBM + Platt) y rechazo *fail-closed* ante checksum corrupto (`test_phase1_onnx_parity_and_p0.py`).
- Invariancia bit-a-bit en Walk-Forward ante *Oracle Future Shock*, `PurgedKFold([t0, t1])` y Hurst sobre retornos estacionarios (`test_phase2_leakage_and_hurst.py`).
- Filtro CUSUM causal, estrategias primarias, simetría Long/Short de `MetaBarrierLabeler` y `sample_weights` (`test_phase3_meta_labeling_and_weights.py`).
- MDA purgado, DSR empírico, CSCV PBO formal ($T \times N$) y flujo end-to-end `MetaModelTrainerV3` → `ExecutionGateway` (`test_phase4_meta_training_mda_dsr_pbo.py`).
- Tick Bars y Tick Imbalance Bars sobre ticks con sonrisa de volumen intradía (`test_phase5_information_bars.py`).
- Señales v3.1: alineación macro estricta `T_macro < T_asset` (`test_macro_rates_causal.py`), filtro de sesión líquida 07–17 UTC + microestructura VPIN (`test_tick_microstructure.py`), primarias económicas (`test_primary_macro_strategies.py`) y torneo pre-registrado $N=6$ con rechazo ante ruido (`test_v31_benchmark_protocol.py`).
- Suite base + v2.0 (`test_trade_vault.py`, `test_cost_sensitive_gatekeeper.py`, `test_regime_detector.py`, `test_macro_rag_agent.py`, `test_onnx_inference.py`, `test_decoupled_architecture.py`, `test_ffd.py`, `test_egarch.py`, `test_risk_manager.py`, `test_triple_barrier.py`, etc.).

### 5.2 Pipeline de Investigación y Entrenamiento (v3.0 / v3.1)
```bash
# 1. Extracción histórica MT5 / macro (y ticks intradía cuando MT5 está conectado)
python src/data_extractor.py

# 2. Preprocesamiento incremental (Indicadores + EGARCH + FFD)
python src/main_preprocessing.py

# 3. Benchmark comparativo OOS: v3.0 (d1_ohlcv) vs v3.1 (d1_macro, N=6)
python src/main_training_v3.py --compare

# 3b. Modos individuales (ej. microestructura TIB cuando existan Parquets de ticks)
python src/main_training_v3.py --mode d1_macro --asset Oro
python src/main_training_v3.py --mode intraday_tib --asset EURUSD

# 4. (Opcional) Re-exportación manual de campeones a ONNX con verificación SHA-256
python src/models/export_champions_to_onnx.py
```

### 5.3 Ejecución en Modo Dual / Gateway / Daemon
```bash
# Modo Dual (Gateway + Daemon MLOps en proceso secundario)
python src/execution/main_bot_v2.py --mode dual --interval 5.0

# Modo Solo Gateway (entorno ligero sin TensorFlow)
python src/execution/main_bot_v2.py --mode gateway --interval 5.0

# Modo Solo Daemon Analítico
python src/execution/main_bot_v2.py --mode daemon
```

---

## 6. Despliegue en Windows Server (AWS EC2 / VPS)

Para operar 24/7 con MetaTrader 5 en una instancia Windows Server (AWS EC2 o VPS):

> **Restricción de GUI de MetaTrader 5:** MetaTrader 5 requiere sesión de escritorio gráfica activa para operar la API IPC. No debe ejecutarse como servicio silencioso de Windows (*Session 0 Isolation*).

1. **Empaquetar artefactos de producción:**
   ```bash
   python export_to_aws.py
   ```
   Genera `bot_production.zip` con `src/`, `.onnx`, `onnx_manifest.json`, `hrp_weights.json`, `requirements/`, `start_bot.bat` y configuraciones de campeones.
2. **Habilitar Auto-Login en Windows Server** (para sobrevivir a reinicios por parches):
   ```cmd
   reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device" /v DevicePasswordLessBuildVersion /t REG_DWORD /d 0 /f
   ```
   Luego ejecutar `netplwiz` (`Win + R`) y desmarcar *"Los usuarios deben escribir su nombre y contraseña para usar el equipo"*.
3. **Crear tarea programada `onlogon`** apuntando a [`start_bot.bat`](start_bot.bat) (espera 60s a que cargue MT5 y reinicia ante caídas):
   ```cmd
   schtasks /create /tn "QuantBot_Trading" /tr "C:\Users\Administrator\Desktop\quant-trading-bot\start_bot.bat" /sc onlogon /ru "Administrator" /rl highest /f
   ```

---

## 7. Auditoría y Mantenimiento MLOps

| Tarea | Frecuencia | Comando | Propósito |
| :--- | :--- | :--- | :--- |
| Refresco de datos | Mensual | `python src/data_extractor.py && python src/main_preprocessing.py` | Actualiza velas MT5 y features causales |
| Reentrenamiento y auditoría v3.0 / v3.1 | Semestral / Anual | `python src/main_training_v3.py --compare` | Compara `d1_ohlcv` vs `d1_macro` ($N=6$), valida compuertas y exporta ONNX |
| Rebalanceo HRP | Mensual | `python src/evaluation/portfolio_backtester.py` | Actualiza `results/hrp_weights.json` |
| Auditoría de trades en vivo | Bajo demanda | `python src/evaluation/live_evaluator.py` | Genera `results/live_production_report.html` |
| Consulta directa SQLite WAL | Bajo demanda | `sqlite3 results/trading_vault.db "SELECT timestamp_utc, symbol, signal_direction, hurdle_ratio, status FROM execution_audit_log ORDER BY id DESC LIMIT 10;"` | Inspección transaccional sin bloquear ejecución |
| Tracking de experimentos | Bajo demanda | `mlflow ui` | Visualización de corridas en `http://127.0.0.1:5000` |

---

## Licencia
- **Autor:** Ronald Solares — Ingeniero Civil Industrial.
- **Licencia:** MIT ([LICENSE](LICENSE)).
