# 🏛️ Especificación Institucional Quant v3.0 — Auditoría Forense Completa & Refactorización Metodológica AFML
## De Prototipo ML Direccional + Infraestructura v2.0 a Sistema de Alpha Institucional (Meta-Labeling de 2 Etapas, CUSUM, Sample Uniqueness, DSR/PBO Formal y Paridad Real ONNX)

> **Estado del Documento:** Especificación Arquitectónica & Auditoría Forense v3.0 (Lista para ejecución SDD/TDD)  
> **Fecha de Auditoría:** 23 de Noviembre de 2026  
> **Alcance:** Auditoría integral de [`quant-trading-bot`](file:///c:/Users/ronal/quant-trading-bot) (`src/preprocessing/`, `src/models/`, `src/evaluation/`, `src/execution/`) + alineación con material de estudio y defensa técnica (`AFML_Resumen.md` y `defensa_data_science_mlops_y_quant.md`).  
> **Marco de Referencia:** Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) & Bailey, Borwein, López de Prado, Zhu (2014), *The Probability of Backtest Overfitting*.

---

## 📑 Índice Ejecutivo

1. [Resumen Ejecutivo y Veredicto de Auditoría](#1-resumen-ejecutivo-y-veredicto-de-auditoría)
2. [Reporte Oficial de Integridad Temporal (Temporal Leakage Guard)](#2-reporte-oficial-de-integridad-temporal-temporal-leakage-guard)
3. [Registro Forense Completo de los 11 Hallazgos (P0 / P1 / P2)](#3-registro-forense-completo-de-los-11-hallazgos-p0--p1--p2)
4. [Clarificación Arquitectónica Crítica: Modelo Primario vs. Meta-Modelo vs. HMM / Autoencoder P90-P99](#4-clarificación-arquitectónica-crítica-modelo-primario-vs-meta-modelo-vs-hmm--autoencoder-p90-p99)
5. [Pilar 1 — Muestreo por Información, Filtro CUSUM Simétrico y Corrección de Hurst (AFML Cap. 2)](#5-pilar-1--muestreo-por-información-filtro-cusum-simétrico-y-corrección-de-hurst-afml-cap-2)
6. [Pilar 2 — Meta-Labeling Verdadero de 2 Etapas y Triple Barrera Direccional (AFML Cap. 3.6)](#6-pilar-2--meta-labeling-verdadero-de-2-etapas-y-triple-barrera-direccional-afml-cap-36)
7. [Pilar 3 — Sample Uniqueness, Purga Exacta por `t1`, Calibración Probabilística y Retiro de Redes Sobreparametrizadas (AFML Cap. 4, 6, 7, 8)](#7-pilar-3--sample-uniqueness-purga-exacta-por-t1-calibración-probabilística-y-retiro-de-redes-sobreparametrizadas-afml-cap-4-6-7-8)
8. [Pilar 4 — Gobernanza Estadística Real: Cero Threshold Snooping, DSR Activo y PBO Multi-Estrategia (AFML Cap. 11–14)](#8-pilar-4--gobernanza-estadística-real-cero-threshold-snooping-dsr-activo-y-pbo-multi-estrategia-afml-cap-1114)
9. [Pilar 5 — Reparación Crítica de Inferencia en Producción v2.0 → v3.0 (ONNX Real y Paridad Train/Serve)](#9-pilar-5--reparación-crítica-de-inferencia-en-producción-v20--v30-onnx-real-y-paridad-trainserve)
10. [Contratos de Datos, Esquema de Artefactos y Plan de Implementación SDD/TDD](#10-contratos-de-datos-esquema-de-artefactos-y-plan-de-implementación-sddtdd)
11. [Matriz de Evolución Completa (v1.0 → v2.0 → v3.0)](#11-matriz-de-evolución-completa-v10--v20--v30)

---

## 1. Resumen Ejecutivo y Veredicto de Auditoría

La evolución **v2.0** resolvió de forma sobresaliente la **capa de ingeniería de sistemas y MLOps**: desacoplamiento de procesos (`ExecutionGateway` vs. `AnalyticsDaemon`), bóveda transaccional ACID in-process (`TradeVault` SQLite WAL), filtro de fricción microestructural (`CostSensitiveGatekeeper` con Hurdle Rate $\ge 2.5\times$), detector HMM 3-estados + KS-Drift y protección pre-noticias (`MacroHazardGuard`).

Sin embargo, una auditoría forense profunda línea por línea sobre la **capa de generación de señales (Alpha), la validación estadística y el puente real de despliegue ONNX** demostró que el proyecto conserva **11 fallas críticas y metodológicas** heredadas de la v1.0 y de la integración v2.0:

1. **La capa de ejecución ONNX v2.0 hoy ejecuta un grafo sintético dummy y tiene bugs silenciosos de integración** (`export_champions_to_onnx.py` no exporta los pesos reales; `execution_gateway.py` omite `import json`, omite el `StandardScaler`, usa `look_back=10` vs `60` y recalcula `d*` de FFD en caliente).
2. **Lo que se llamó "Meta-Labeling" en v1.0/v2.0 es en realidad Triple Barrier Labeling unidireccional directo**, sin Modelo Primario ni Filtro CUSUM, con un error de asimetría matemática al invertir `1 - P(Long)` para abrir Shorts en `EURUSD`.
3. **Existen fugas temporales reales (Data Leakage)** en el reentrenamiento Walk-Forward (`y_test[:i+1]` sin purgar las últimas `max_hold=10` etiquetas aún abiertas), en el `StandardScaler` pre-CV de LSTM/BiLSTM, en `cpcv_auditor.py` (multiplicación contemporánea `daily_returns * signals` sin `shift(1)`), en el uso de `.bfill()`, y en la optimización de `confidence_threshold` directamente sobre el OOS (`df_backtest`).
4. **Sobreparametrización y Multiple Testing sin penalización:** Dos campeones (`Oro` y `SP500`) son BiLSTMs de ~300K–1.2M parámetros entrenadas sobre ~2,500–4,000 velas diarias (en `SP500` con solo 2 features) elegidas como el máximo de $6 \times 6 \times 6 = 216$ combinaciones con el filtro DSR desactivado.

**Conclusión:** La arquitectura de ejecución v2.0 **se preserva intacta como chasis institucional**, mientras que esta especificación **v3.0** reconstruye con rigor matemático de Hedge Fund la capa de datos/eventos, el Meta-Labeling verdadero de 2 etapas, la validación anti-overfitting y la paridad exacta Train/Serve en ONNX.

---

## 2. Reporte Oficial de Integridad Temporal (Temporal Leakage Guard)

### ⏱️ Temporal Leakage Audit Report (`quant-trading-bot` Estado v2.0)

- **Scaler / Imputer Isolation:** ⚠️ **Leakage Detected**
  - [`src/models/bilstm_model.py#L100-L103`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L100-L103), [`src/models/lstm_model.py#L91-L94`](file:///c:/Users/ronal/quant-trading-bot/src/models/lstm_model.py#L91-L94) y [`src/models/arima_lstm.py#L127-L131`](file:///c:/Users/ronal/quant-trading-bot/src/models/arima_lstm.py#L127-L131) ejecutan `StandardScaler().fit_transform(X_train)` **antes** de partir en folds de Purged CV (media y desviación de los folds de validación filtran al entrenamiento).
  - [`src/data_extractor.py#L121-L127`](file:///c:/Users/ronal/quant-trading-bot/src/data_extractor.py#L121-L127) y `prepare_sequences()` en `xgb_model.py#L33`, `random_forest.py#L33`, `bilstm_model.py#L38`, `lstm_model.py#L37` aplican `.bfill()`, rellenando valores pasados faltantes con observaciones futuras.
- **Target & Feature Lags:** ⚠️ **Contemporaneous & Horizon Leakage Detected**
  - **Walk-Forward Retrain Leakage:** [`xgb_model.py#L154-L157`](file:///c:/Users/ronal/quant-trading-bot/src/models/xgb_model.py#L154-L157), [`random_forest.py#L158-L161`](file:///c:/Users/ronal/quant-trading-bot/src/models/random_forest.py#L158-L161), [`bilstm_model.py#L208-L213`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L208-L213) y [`lstm_model.py#L199-L204`](file:///c:/Users/ronal/quant-trading-bot/src/models/lstm_model.py#L199-L204) reentrenan en el paso `i` con `y_test[:i+1]`. Dado que cada etiqueta Triple Barrier depende de `high`/`low` hasta `i + max_hold` (`+10` días), las últimas 10 etiquetas no existen en el instante `i` y filtran resultados futuros hacia el predictor del paso `i+1`.
  - **CPCV Auditor 1-Bar Lookahead:** [`src/evaluation/cpcv_auditor.py#L98-L102`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/cpcv_auditor.py#L98-L102) multiplica `daily_returns[i]` (retorno hasta `close[i]`) por `signals[i]` sin `shift(1)`.
- **Split Chronology:** ⚠️ **Test-Set Snooping & Sequence Shuffling**
  - [`src/evaluation/alpha_backtester.py#L482`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/alpha_backtester.py#L482) recorre `for umbral in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]` directamente sobre el conjunto de evaluación OOS (`df_backtest`) y elige el umbral ganador sobre ese mismo OOS.
  - Los entrenadores Keras usan `shuffle=True` sobre ventanas de 60 días que comparten 59/60 pasos y etiquetas solapadas.
- **Execution / Fill Realism:** ✅ **Backtester Causal** (`opens[i+1]` + heurística pesimista si toca TP y SL el mismo día en [`alpha_backtester.py#L246-L298`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/alpha_backtester.py#L246-L298)) | ⚠️ **Paridad Rota en Producción v2.0** (`export_champions_to_onnx.py` genera un grafo dummy; `execution_gateway.py` omite `import json`, no escala features, usa `look_back=10` y recalcula FFD `d*` en vivo).

👉 **Summary & Fix:** Purgar toda etiqueta cuyo $t_1 \ge T_{\text{now}}$ en CV y Walk-Forward; mover `StandardScaler` dentro de un `sklearn.pipeline.Pipeline` ajustado únicamente por fold; eliminar `.bfill()`; optimizar umbrales exclusivamente en validación interna; exportar el pipeline real a ONNX y congelar `d*` de FFD en el artefacto del campeón.

---

## 3. Registro Forense Completo de los 11 Hallazgos (P0 / P1 / P2)

| ID | Severidad | Componente / Archivo Exacto | Defecto Verificado en Código | Impacto Real |
| :---: | :---: | :--- | :--- | :--- |
| **F-01** | 🔴 **P0** | [`export_champions_to_onnx.py#L44`](file:///c:/Users/ronal/quant-trading-bot/src/models/export_champions_to_onnx.py#L44)<br>[`onnx_exporter.py#L26-L110`](file:///c:/Users/ronal/quant-trading-bot/src/models/onnx_exporter.py#L26-L110) | Llama a `create_synthetic_onnx_model()` con pesos constantes `W = 0.1/d, B = 0` en lugar de convertir los modelos `.pkl`/`.keras` reales. | **Producción inválida:** el gateway ONNX no ejecuta el modelo entrenado sino una regresión lineal dummy. |
| **F-02** | 🔴 **P0** | [`execution_gateway.py#L118`](file:///c:/Users/ronal/quant-trading-bot/src/execution/execution_gateway.py#L118), `#L226`, `#L237`, `#L244`<br>[`risk_manager.py#L93`](file:///c:/Users/ronal/quant-trading-bot/src/execution/risk_manager.py#L93)<br>[`analytics_daemon.py#L85`](file:///c:/Users/ronal/quant-trading-bot/src/execution/analytics_daemon.py#L85) | (a) Falta `import json` → `NameError` silenciado deja `onnx_engine = None` (`prob = 0.50`). (b) Features sin `StandardScaler` y `look_back=10` vs `60`. (c) FFD recalcula `d*` en 300 barras vivas. (d) `HybridRiskMonitor()` sin `data_dir` deja HMM/KS en `None`. (e) `AnalyticsDaemon` pasa shape `(30, 1)` al Autoencoder entrenado en `(N, 3)`. | El gateway v2.0 queda ciego o descalibrado al operar en vivo y los monitores fallan silenciosamente. |
| **F-03** | 🔴 **P0** | [`xgb_model.py#L155`](file:///c:/Users/ronal/quant-trading-bot/src/models/xgb_model.py#L155), [`random_forest.py#L159`](file:///c:/Users/ronal/quant-trading-bot/src/models/random_forest.py#L159), [`bilstm_model.py#L211`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L211), [`lstm_model.py#L202`](file:///c:/Users/ronal/quant-trading-bot/src/models/lstm_model.py#L202) | Reentrenamiento Walk-Forward incluye `y_test[:i+1]` sin purgar los últimos `max_hold=10` días cuyas barreras aún no han cerrado en el día `i`. | **Data Leakage real** en backtest Walk-Forward: el modelo ve resoluciones futuras de barrera. |
| **F-04** | 🔴 **P0** | [`meta_labeling.py#L41-L78`](file:///c:/Users/ronal/quant-trading-bot/src/preprocessing/meta_labeling.py#L41-L78)<br>[`alpha_backtester.py#L241-L243`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/alpha_backtester.py#L241-L243)<br>[`campeon_EURUSD.json#L12`](file:///c:/Users/ronal/quant-trading-bot/results/campeon_EURUSD.json#L12) | Etiquetado Long-only incondicional cada día (sin modelo primario ni CUSUM). En `EURUSD` (`bilateral=True`, `threshold=0.50`), `prob < 0.50` abre Short tratando `1 - P(Long TP)` como `P(Short TP)` pese a barreras asimétricas (`2.0σ` vs `1.5σ`) y expiraciones planas en `Label=0`. | No es Meta-Labeling real; en `EURUSD` fuerza pronóstico direccional continuo con sesgo matemático. |
| **F-05** | 🟠 **P1** | [`alpha_backtester.py#L482-L574`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/alpha_backtester.py#L482-L574)<br>[`portfolio_backtester.py#L33-L86`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/portfolio_backtester.py#L33-L86) | Grid search de 6 umbrales directamente sobre el Test Set OOS (`df_backtest`) + selección entre 36 modelos × 6 umbrales = 216 pruebas **sin filtro DSR** (y `std_sharpes = 0.5` hardcodeado). | **Test-Set Snooping** y **Selection Bias** inflan falsamente Alpha/CAGR de modelos de alta varianza. |
| **F-06** | 🟠 **P1** | [`bilstm_model.py#L68-L86`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L68-L86)<br>[`campeon_SP500.json`](file:///c:/Users/ronal/quant-trading-bot/results/campeon_SP500.json), [`campeon_Oro.json`](file:///c:/Users/ronal/quant-trading-bot/results/campeon_Oro.json) | BiLSTM apilada de 2 capas (~300K–1.2M parámetros) sobre ~2,500–4,000 velas diarias (`SP500` con solo 2 features: `VIX`, `DXY`), sin `EarlyStopping` en entrenamiento maestro (`epochs=20`, `shuffle=True`). | Relación parámetros/muestras > 100:1; sobreajuste a ruido y beta alcista en `bilateral=False`. |
| **F-07** | 🟠 **P1** | [`cpcv_auditor.py#L97-L115`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/cpcv_auditor.py#L97-L115) | (a) Multiplica `daily_returns * signals` sin `shift(1)` (lookahead de 1 vela). (b) Evalúa 1 solo modelo y calcula `% de paths con Sharpe <= 0` en vez del PBO formal de Bailey & López de Prado (matriz $T \times N$). | El certificado `PBO < 5%` actual **no mide PBO formal** y está sesgado al alza por el lookahead de 1 barra. |
| **F-08** | 🟠 **P1** | [`xgb_model.py#L40-L43`](file:///c:/Users/ronal/quant-trading-bot/src/models/xgb_model.py#L40-L43), [`random_forest.py#L40-L43`](file:///c:/Users/ronal/quant-trading-bot/src/models/random_forest.py#L40-L43) | Sin pesos de unicidad de muestra (AFML Cap. 4) pese a concurrencia $c_t \approx 10$; además aplana `60 × F` (hasta 480 columnas crudas altamente colineales) para árboles de decisión. | Violación de IID, árboles sobreajustados a rezagos ruidosos y varianza subestimada. |
| **F-09** | 🟠 **P1** | [`bilstm_model.py#L100`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L100), [`data_extractor.py#L121`](file:///c:/Users/ronal/quant-trading-bot/src/data_extractor.py#L121), [`asset_screener.py#L159`](file:///c:/Users/ronal/quant-trading-bot/src/preprocessing/asset_screener.py#L159) | (a) `StandardScaler` ajustado antes de los folds CV en redes neuronales. (b) `.bfill()` no causal. (c) Hurst R/S calculado sobre precios crudos $I(1)$ (`closes`) en vez de retornos estacionarios o Variance Ratio. | Micro-leakage en CV y estimador de Hurst inflado artificialmente ($H \approx 0.7\text{–}0.9$ sobre niveles de precio). |
| **F-10** | 🟡 **P2** | [`xgb_model.py#L91`](file:///c:/Users/ronal/quant-trading-bot/src/models/xgb_model.py#L91), [`bilstm_model.py#L123`](file:///c:/Users/ronal/quant-trading-bot/src/models/bilstm_model.py#L123) | Optuna maximiza `accuracy` / `f1_macro` duro en `0.50` y no calibra probabilidades (Platt / Isotonic). | Las salidas $\hat{p}$ no son probabilidades calibradas, distorsionando $E[U]$ del `CostSensitiveGatekeeper` y el **Criterio de Kelly**. |
| **F-11** | 🟡 **P2** | [`data_extractor.py#L258-L265`](file:///c:/Users/ronal/quant-trading-bot/src/data_extractor.py#L258-L265) | Muestreo exclusivo por Time Bars (`D1` / `H4`) y selección de features por MDI in-sample sin **MDA Out-of-Sample** (AFML Cap. 8). | En `H4` las sesiones muertas (Asia) pesan igual que NY/Londres; features ruidosas no se podan sistemáticamente. |

---

## 4. Clarificación Arquitectónica Crítica: Modelo Primario vs. Meta-Modelo vs. HMM / Autoencoder P90-P99

Una confusión frecuente al integrar sistemas MLOps con López de Prado es asumir que el **HMM 3-Estados** o el **LSTM Autoencoder (P90/P99)** cumplen el rol de *"Modelo Primario"* del Meta-Labeling. **No lo son.** Un sistema cuantitativo institucional separa estrictamente **4 capas ortogonales**:

```text
═══════════════════════════════════════════════════════════════════════════════════════════════
              ARQUITECTURA CUANTITATIVA DE 4 CAPAS ORTOGONALES (ALPHAEDGE v3.0)
═══════════════════════════════════════════════════════════════════════════════════════════════

 [CAPA 0: MUESTREO DE INFORMACIÓN Y EVENTOS — AFML Cap. 2]
   • D1: Filtro CUSUM Simétrico calibrado por volatilidad EGARCH (solo dispara en movimientos estructurales)
   • Intradía (H4/H1 opcional): Tick Imbalance Bars / Tick Volume Bars desde mt5.copy_ticks_range()
   • Salida: Subconjunto disperso de timestamps de eventos {t_0} (elimina ruido de días planos)
                                          │
                                          ▼
 [CAPA 1: MODELO PRIMARIO (GENERADOR DE LADO / HIPÓTESIS ECONÓMICA) — AFML Cap. 3.6]
   • Estrategia estructural transparente y de ALTO RECALL (0 redes neuronales):
       - Tendencia / Breakout (Donchian / EMA-Fast vs EMA-Slow / TSMOM)
       - O Reversión en rango según régimen del activo (Hurst sobre retornos)
   • Salida: Dirección candidata s_{t0} ∈ {+1 (LONG), -1 (SHORT)} — NUNCA decide el tamaño
                                          │
                                          ▼
 [CAPA 2: META-MODELO ML CALIBRADO (FILTRO DE CALIDAD & BET SIZING) — AFML Cap. 3.6, 4, 6, 8, 10]
   • Recibe ÚNICAMENTE los eventos {t_0} con su lado s_{t0} + features tabulares resumidas
     (FFD congelado, EGARCH, vol ratios, macro VIX/DXY/Yield10Y, estado HMM como feature contextual)
   • Modelo: GBDT Regularizado (XGBoost / LightGBM) o Logistic L2 / Random Forest (max_samples = ū)
     entrenado con Sample Uniqueness Weights, Purga exacta por [t0, t1], optimizando Log-Loss + Calibración
   • Salida: Probabilidad CALIBRADA p̂ = P(TP antes que SL y E[r_net] > 0 | evento t_0, lado s_{t0})
   • Si p̂ < θ* (calibrado en CV interno): ABORTAR (drop trade débil → sube Precision drásticamente)
                                          │
                                          ▼
 [CAPA 3: RISK OVERLAYS & GOBERNANZA MLOps (NO DECIDEN DIRECCIÓN) — Infraestructura v2.0]
   1. MacroHazardGuard:       Freeze pre-noticias (-30m / +15m FOMC, CPI, NFP)
   2. CostSensitiveGatekeeper: E[U(p̂)] = p̂·R_TP - (1-p̂)·|R_SL| - C_fricción >= 2.5 · C_fricción
   3. HMM 3-Estados + KS-Drift: Modulador de exposición (Choppy → 0.25x, KS p < 0.01 → 0.50x)
   4. Autoencoder P90 / P99:   Monitor forense sobre telemetría de TRADES REALIZADOS [net_ret, dur, win]
                               (P90 = alerta Concept Drift; P99 + DD = cuarentena O(1) en TradeVault)
   5. HRP + Fractional Kelly:  Asignación de portafolio y tamaño de lote → MT5 + SQLite WAL ACID
═══════════════════════════════════════════════════════════════════════════════════════════════
```

### ¿Por qué HMM y Autoencoder P90/P99 NO son Modelo Primario?

| Módulo | Qué observa | Qué decide | Rol Correcto |
| :--- | :--- | :--- | :--- |
| **Modelo Primario** | Estructura de precio / momentum / breakout del activo | **Dirección (`+1` / `-1`) y momento de entrada (`t0`)** | Generador de Hipótesis (Capa 1) |
| **Meta-Modelo ML** | Contexto FFD, EGARCH, macro (`VIX`, `DXY`, `Yield10Y`), régimen HMM, volatilidad | **Si tomar o descartar la señal primaria y con qué probabilidad $\hat{p}$** | Filtro de Precisión y Sizing (Capa 2) |
| **HMM 3-Estados** | Retornos, volatilidad y volumen recientes | **Régimen ambiental** (`Bull` / `Bear` / `Choppy`) → factor de escala (`1.0x` vs `0.25x`) | Risk Overlay + Feature Contextual (Capa 3) |
| **Autoencoder P90/P99** | Rendimiento **ex-post** de los últimos 10–30 trades cerrados (`[net_ret, duration, win]`) | **Salud de la estrategia** → alerta Drift (P90) o Cuarentena/Kill-Switch (P99) | Circuit Breaker MLOps (Capa 3) |

> **Nota Clave de Diseño:** El HMM sí puede inyectarse como **feature categórica/probabilidad posterior** `P(state = k | X_{1:t})` dentro del Meta-Modelo (Capa 2) —siempre usando *filtered probabilities* (algoritmo Forward, **nunca** *smoothed probabilities* Viterbi/Forward-Backward que miran el futuro!)— además de actuar como overlay de Kelly en Capa 3.

---

## 5. Pilar 1 — Muestreo por Información, Filtro CUSUM Simétrico y Corrección de Hurst (AFML Cap. 2)

### 5.1 Política de Barras según Horizonte Operativo
1. **Swing Diario (`D1` — Producción Actual):**
   - Mantener velas diarias **D1** como unidad base de consolidación macro (ya agrega toda la sonrisa de volumen intradía y alinea limpiamente con `VIX`, `DXY` y `Yield10Y`), **PERO eliminar el muestreo uniforme de todos los días** aplicando un **Filtro CUSUM Simétrico dinámico** (AFML §2.5.2.1).
   - **Definición Matemática del Filtro CUSUM Simétrico:**
     Sea $r_t = \ln(C_t / C_{t-1})$ el retorno logarítmico diario y $h_t = \kappa_{\text{cusum}} \cdot \sigma_{t-1,\text{EGARCH}}$ el umbral dinámico (ej. $\kappa_{\text{cusum}} \in [0.8, 1.5]$):
     $$S_t^+ = \max\!\left(0,\; S_{t-1}^+ + r_t\right), \qquad S_t^- = \min\!\left(0,\; S_{t-1}^- + r_t\right)$$
     Se emite un evento candidato $t_0 = t$ si y solo si $S_t^+ \ge h_t$ o $S_t^- \le -h_t$, reiniciando el acumulador correspondiente ($S_t^+ \leftarrow 0$ o $S_t^- \leftarrow 0$).
   - **Efecto Directo:** Reduce el dataset de ~3,500 filas altamente redundantes a ~600–1,000 **eventos con verdadera llegada de información**, reduciendo solapamiento de etiquetas y cortando el entrenamiento sobre días muertos laterales.
2. **Intradía (`H4` / `H1` — Solo si se habilita):**
   - Prohibido entrenar modelos ML sobre Time Bars `H4`/`H1` crudas (sesgo severo Asia vs. Londres/NY).
   - Como MT5 OTC devuelve `real_volume == 0` en Forex/CFDs, construir **Tick Bars** (cada $M$ ticks) o **Tick Imbalance Bars (TIBs)** desde `mt5.copy_ticks_range()` (usando tick rule $b_t = \text{sign}(\Delta p_t)$) antes de extraer features y aplicar CUSUM.

### 5.2 Corrección del Estimador de Hurst (`src/preprocessing/asset_screener.py`)
- **Bug Actual (F-09):** `AssetScreener.calculate_hurst_exponent(closes)` aplica R/S sobre **niveles de precio $I(1)$**, produciendo valores espurios $H \approx 0.70\text{–}0.90$ incluso para un random walk puro.
- **Corrección v3.0:**
  1. Calcular **Log-Returns** $r_t = \ln(C_t / C_{t-1})$ estacionarios y aplicar **Lo's Modified R/S** o **Variance Ratio Test** de Lo-MacKinlay ($VR(q) = \frac{\text{Var}(r_t^{(q)})}{q\,\text{Var}(r_t)}$ con estadístico heterocedástico $z^*(q)$).
  2. Mantener **Detrended Fluctuation Analysis (DFA)** o `hurst(log_returns)` con validación Monte Carlo contra ruido blanco simulado (para verificar empíricamente que un Random Walk arroja $H \approx 0.50 \pm 0.03$ en test unitario!).

### 5.3 Congelamiento Determinista de FFD (`src/preprocessing/stationarity.py`)
- Eliminar `adfuller` sobre la serie completa (evaluar solo sobre `train_slice`).
- Persistir el diccionario exacto `ffd_d_map: Dict[str, float]` (ej. `{"open": 0.35, "high": 0.35, "low": 0.35, "DXY_close": 0.42}`) junto al modelo campeón.
- En producción (`ExecutionGateway`), `FractionalDifferencer.transform_with_fixed_d(df, d_map)` aplica **exactamente los mismos pesos $\omega_k(d^*)$** sin recalcular `d*` nunca en vivo.
- **Prohibición Absoluta de `.bfill()`:** Reemplazar todo `.ffill().bfill()` por `.ffill()` seguido de `.dropna()` explícito del periodo de calentamiento (*warmup*).

---

## 6. Pilar 2 — Meta-Labeling Verdadero de 2 Etapas y Triple Barrera Direccional (AFML Cap. 3.6)

### 6.1 Nuevo Módulo: `src/strategies/primary_strategies.py`
Separar físicamente la **hipótesis de dirección** del **clasificador ML**. Cada estrategia primaria hereda de `BasePrimaryStrategy` y retorna una serie causal `side` $\in \{-1, 0, +1\}$ (disponible estrictamente al cierre de barra $t_0$ para entrar en $\text{Open}_{t_0+1}$):

| Estrategia Primaria | Hipótesis Económica | Regla Causal (Alto Recall) | Activos Objetivo |
| :--- | :--- | :--- | :--- |
| **`TrendPullbackPrimary`** | Prima de momentum tendencial + entrada en retroceso | Régimen estructural `Close > SMA(200)` → `+1` en evento CUSUM o cruce `EMA(20) > EMA(50)`; si `bilateral=True` y `Close < SMA(200)` → `-1`. | `Oro`, `SP500`, `EURUSD` |
| **`DonchianBreakoutPrimary`** | Ruptura de liquidez / consolidación | `Close > max(High[t-20:t-1])` → `+1`; `Close < min(Low[t-20:t-1])` → `-1` (si bilateral). | `EURUSD`, `Oro` |
| **`VolatilityMeanReversionPrimary`** | Sobrerreacción de corto plazo dentro de tendencia | Z-score de retorno 5d `< -1.5` con `Close > SMA(200)` → `+1`; `> +1.5` con `Close < SMA(200)` → `-1`. | Activos con $VR(q) < 1$ |

> **Regla de Oro:** El Modelo Primario tiene **cero hiperparámetros optimizados por gradiente** y **cero redes neuronales**. Debe ser explicable en 1 frase ante un Comité de Inversiones.

### 6.2 Refactorización de `src/preprocessing/meta_labeling.py` → `MetaBarrierLabeler`
Dada la tabla de eventos filtrados por CUSUM con dirección primaria $s_{t_0} \in \{+1, -1\}$:

1. **Entrada Realista:** $P_{\text{entry}} = \text{Open}_{t_0 + 1}$.
2. **Barreras Orientadas por `side` (resuelve F-04 de raíz):**
   $$\text{TP}_{t_0} = P_{\text{entry}} \cdot \left(1 + s_{t_0} \cdot k_{\text{up}} \cdot \frac{\sigma_{t_0}}{100}\right), \qquad \text{SL}_{t_0} = P_{\text{entry}} \cdot \left(1 - s_{t_0} \cdot k_{\text{down}} \cdot \frac{\sigma_{t_0}}{100}\right)$$
   - Si $s_{t_0} = +1$: TP toca cuando `High >= TP`, SL toca cuando `Low <= SL`.
   - Si $s_{t_0} = -1$: TP toca cuando `Low <= TP`, SL toca cuando `High >= SL`.
   - Si ambos tocan en la misma vela: **heurística pesimista** (asume SL).
3. **Etiqueta Meta Binaria Neta de Fricción:**
   Se calcula el retorno neto descontando spread/slippage estimado $c_{\text{rt}}$:
   $$r^{\text{net}}_{t_0, t_1} = s_{t_0} \cdot \left(\frac{P_{\text{exit}}}{P_{\text{entry}}} - 1\right) - c_{\text{rt}}$$
   $$y^{\text{meta}}_{t_0} = \begin{cases} 1 & \text{si } r^{\text{net}}_{t_0, t_1} > 0 \text{ (o primer toque = TP neto positivo)} \\ 0 & \text{en caso contrario (SL o expiración negativa/plana)} \end{cases}$$
4. **Esquema Obligatorio de Salida (`events_df`):**
   Cada fila del dataset de entrenamiento del Meta-Modelo conserva su **intervalo exacto de vida**:
   - `t0` (`datetime`): timestamp de la señal al cierre.
   - `t1` (`datetime`): timestamp exacto de cierre de barrera (TP, SL o barrera vertical `max_hold`).
   - `side` (`int8`): `+1` o `-1` (entregado también como feature al Meta-Modelo!).
   - `meta_label` (`int8`): `1` (tomar el trade) o `0` (descartar el trade).
   - `net_ret` (`float64`): retorno neto realizado del evento.

**Impacto Inmediato en `EURUSD`:** Se elimina para siempre el error `prob < 1 - threshold → Short`. Ahora el Modelo Primario propone `side = -1` (Short) y el Meta-Modelo evalúa directamente $\hat{p} = P(\text{este Short gana netamente} \mid X_{t_0}, \text{side}=-1)$. Si $\hat{p} < \theta^*$, el bot pasa a **CASH** (filtra de verdad).

---

## 7. Pilar 3 — Sample Uniqueness, Purga Exacta por `t1`, Calibración Probabilística y Retiro de Redes Sobreparametrizadas (AFML Cap. 4, 6, 7, 8)

### 7.1 Pesos por Unicidad de Muestra y Decaimiento Temporal (`src/preprocessing/sample_weights.py`)
Para corregir la violación de IID en eventos concurrentes (AFML Cap. 4):

1. **Concurrencia en cada barra $t$:**
   $$c_t = \sum_{i=1}^{I} \mathbb{I}(t_{0,i} \le t \le t_{1,i})$$
2. **Unicidad Media del evento $i$:**
   $$\bar{u}_i = \frac{1}{|[t_{0,i}, t_{1,i}]|} \sum_{t=t_{0,i}}^{t_{1,i}} \frac{1}{c_t}$$
3. **Atribución por Retorno y Time-Decay:**
   $$\tilde{w}_i = \left|\sum_{t=t_{0,i}}^{t_{1,i}} \frac{r_t}{c_t}\right| \cdot d_i, \qquad w_i = I \cdot \frac{\tilde{w}_i}{\sum_{j=1}^I \tilde{w}_j}$$
   con decaimiento lineal conservador $d(c_{\text{decay}}=0.5)$ que da mayor peso a regímenes recientes sin olvidar crisis pasadas.
4. **Aplicación obligatoria:**
   - Pasar `sample_weight=w_train` en `.fit()` de `XGBClassifier`, `LGBMClassifier`, `LogisticRegression` y `RandomForestClassifier`.
   - En `RandomForestClassifier`, fijar `max_samples = float(np.clip(avg_uniqueness, 0.1, 1.0))` (aproximación computacionalmente eficiente al *Sequential Bootstrapping* de AFML §4.5.3).

### 7.2 Purga Exacta por `[t0, t1]` en CV y Walk-Forward (`src/validation/purged_cv.py`)
1. **En Purged K-Fold / CPCV:** Dado un fold de prueba con intervalo $[T_{\text{val,start}}, T_{\text{val,end}}]$:
   - **Purge:** Eliminar todo evento de entrenamiento $i$ tal que $t_{1,i} \ge T_{\text{val,start}}$ y $t_{0,i} \le T_{\text{val,end}}$.
   - **Embargo:** Eliminar todo evento de entrenamiento $i$ tal que $T_{\text{val,end}} < t_{0,i} \le T_{\text{val,end}} + \Delta_{\text{embargo}}$ ($\Delta_{\text{embargo}} = 1\%\text{–}2\%$ del número de barras).
2. **En Walk-Forward Retraining (Arregla F-03):**
   Al reentrenar en el instante $T_{\text{now}}$:
   ```python
   # ESTRICTAMENTE CAUSAL: Solo trades cuya barrera ya se resolvió ANTES de T_now
   known_mask = events_df["t1"] < current_time
   ```
   Cero etiquetas abiertas en el instante de reentrenamiento.
3. **Aislamiento de Preprocesamiento (Arregla F-09):**
   Todo `StandardScaler` y `SimpleImputer` vive dentro de un `sklearn.pipeline.Pipeline` que se ajusta (`fit`) exclusivamente sobre `train_idx` dentro de cada fold.

### 7.3 Ingeniería Tabular Multi-Escala en vez de Ventanas Aplanadas de 480 Columnas
Reemplazar el aplanado bruto `window = dataset[i-60:i, :].flatten()` (480 columnas colineales) por **15–25 features tabulares de alta relación señal/ruido** por evento $t_0$:
- **Estructura & Memoria:** `close_FFD`, `DXY_close_FFD`, `Yield10Y_FFD`, distancia normalizada a `SMA(50)` y `SMA(200)` en unidades de ATR: `(Close - SMA_200) / ATR_14`.
- **Volatilidad & Microestructura:** `EGARCH_Vol`, ratio de compresión de volatilidad `ATR_14 / ATR_60`, `VIX_close`, cambio 5d y 20d de `VIX`, rango relativo `(High - Low) / ATR_14`, z-score de `tick_volume` (ventana 20d causal).
- **Momentum Multi-Escala:** retornos acumulados normalizados por volatilidad a 5d, 20d, 60d; `RSI_14`; pendiente de `MACD_Hist`.
- **Contexto del Evento:** `primary_side` (`+1` / `-1`), `hmm_filtered_prob_choppy` (probabilidad filtrada causal del HMM, **NO Viterbi smoothed**), días desde último evento CUSUM.

### 7.4 Selección de Familias de Modelos, Calibración Probabilística y MDA (AFML Cap. 6 & 8)
- **Jerarquía de Modelos en v3.0 (Tabular-First para D1):**
  1. `LogisticRegression` (L2 / ElasticNet — baseline lineal indispensable; si una red compleja no vence a una logística regularizada en OOS, se descarta).
  2. `XGBoost` regularizado (`max_depth ∈ [2, 5]`, `min_child_weight ∈ [10, 50]`, `reg_alpha`, `reg_lambda`, `colsample_bytree ∈ [0.5, 0.8]`, `subsample ∈ [0.6, 0.8]`).
  3. `LightGBM` regularizado (`num_leaves ∈ [7, 31]`, `min_child_samples ∈ [20, 80]`).
  4. `RandomForest` con `max_depth ∈ [3, 8]`, `min_samples_leaf ≥ 0.02 * N`, `max_samples = avg_uniqueness`.
  5. **Redes Neuronales (`BiLSTM` / `LSTM` / `ARIMA-LSTM`):** **Retiradas de producción D1** salvo que se restrinjan a arquitecturas ultra-compactas (< 5,000 parámetros, ej. `1D-CNN` o `GRU(16)` con fuerte L2 + Dropout + EarlyStopping en validación purgada) **y** superen estadísticamente a los modelos tabulares tras penalización DSR.
- **Métrica de Optuna y Calibración (Arregla F-10):**
  - Optimizar **`neg_log_loss`** ponderado por `sample_weight` (regla de puntuación estrictamente propia), nunca `accuracy` ni `f1_macro` duro en `0.50`.
  - Aplicar calibración **Platt Scaling (`method='sigmoid'`)** sobre probabilidades out-of-fold purgadas para garantizar que $\hat{p} = 0.65$ corresponda empíricamente a ~65% de aciertos reales, asegurando validez matemática en `CostSensitiveGatekeeper` y **Kelly Sizing**.
  - Medir **Brier Score** y **Expected Calibration Error (ECE < 0.05)** como gate obligatorio antes de promoción a producción.
- **Poda de Features por MDA Out-of-Sample (`src/evaluation/feature_importance.py` — AFML Cap. 8):**
  - Implementar **Mean Decrease Accuracy / Log-Loss (MDA)** permutando cada columna (o cluster colineal vía Clustered MDA) exclusivamente en los folds de validación purgados.
  - Eliminar automáticamente toda feature con $\Delta \text{LogLoss}_{\text{MDA}} \le 0$ (features que añaden ruido puro).

---

## 8. Pilar 4 — Gobernanza Estadística Real: Cero Threshold Snooping, DSR Activo y PBO Multi-Estrategia (AFML Cap. 11–14)

### 8.1 Eliminación Total del Threshold Snooping (Arregla F-05)
- **Prohibido** recorrer `for umbral in [0.50..0.75]` sobre `df_backtest` (Test OOS).
- El umbral de corte $\theta^*$ se deriva **analíticamente o por validación interna**:
  1. **Método Analítico (Preferido):** No usar umbral arbitrario fijo en el Meta-Modelo; delegar el corte directamente al **`CostSensitiveGatekeeper`** usando el punto de equilibrio económico real:
     $$\hat{p}_{\min}(t) = \frac{|R_{SL}| + 2.5 \cdot \mathcal{C}_{\text{fricción}}(t)}{R_{TP} + |R_{SL}|}$$
     *(Si el Meta-Modelo está calibrado en Log-Loss, el umbral óptimo depende del spread y la volatilidad del día, no de un número mágico sobreajustado en backtest).*
  2. **Si se usa umbral adicional $\theta_{\text{floor}}$:** Se optimiza exclusivamente en los folds internos de validación purgada (Nested CV), **congelado** antes de tocar el Test Set OOS.

### 8.2 Deflated Sharpe Ratio (DSR) como Gatekeeper Activo de Selección
En [`portfolio_backtester.py`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/portfolio_backtester.py) y [`alpha_backtester.py`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/alpha_backtester.py):
- Calcular la varianza empírica real $V[\{\widehat{SR}_n\}_{n=1}^N]$ sobre el conjunto efectivo de $N$ estrategias evaluadas (eliminando el `std_sharpes = 0.5` hardcodeado).
- Calcular el umbral de Sharpe esperado por puro azar bajo múltiples pruebas:
  $$\widehat{SR}_0 = \sqrt{V[\{\widehat{SR}_n\}]} \left[ (1-\gamma)\,\Phi^{-1}\!\left(1-\frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1-\frac{1}{N}e^{-1}\right) \right]$$
  donde $\gamma \approx 0.5772156649$ es la constante de Euler-Mascheroni.
- **Compuerta Obligatoria de Promoción:** Ningún modelo puede ser exportado como `campeon_{symbol}.json` si $\text{DSR} < 0.95$ (o fallback documentado a `CASH` / `SMA-200` benchmark si ningún modelo supera la prueba estadística!).

### 8.3 PBO Formal Multi-Estrategia (CSCV — Bailey & López de Prado, 2014) (Arregla F-07)
Refactorizar [`src/evaluation/cpcv_auditor.py`](file:///c:/Users/ronal/quant-trading-bot/src/evaluation/cpcv_auditor.py):
1. **Corregir inmediatamente el lookahead de 1 barra:** Usar retornos realizados por trade del backtester causal o aplicar estrictamente `signals.shift(1) * daily_returns`.
2. **Implementar el algoritmo CSCV (Combinatorially Symmetric Cross-Validation) real:**
   - Entrada: Matriz $M \in \mathbb{R}^{T \times N}$ donde cada columna $n \in \{1,\dots,N\}$ es la serie de retornos diarios netos de una configuración del torneo.
   - Particionar $T$ en $S$ bloques contiguos iguales (par, ej. $S = 10$ o $S = 12$ → $\binom{10}{5} = 252$ splits simétricos; aplicar purga de `max_hold` en las fronteras).
   - Para cada combinación $c$: formar $J$ (In-Sample, $S/2$ bloques) y $\bar{J}$ (Out-Of-Sample, $S/2$ bloques complementarios).
   - Encontrar $n^* = \arg\max_n SR(J_{:,n})$.
   - Calcular el rango relativo de $n^*$ en OOS: $\bar{\omega}_c = \frac{\text{rank}(SR(\bar{J}_{:,n^*}))}{N + 1} \in (0, 1)$ y su logit $\lambda_c = \ln\!\left(\frac{\bar{\omega}_c}{1 - \bar{\omega}_c}\right)$.
   - **PBO Formal:** $\text{PBO} = \frac{1}{|C|} \sum_{c \in C} \mathbb{I}(\lambda_c < 0)$.
   - Reportar también **Stochastic Dominance OOS** y **Degradación de Sharpe IS → OOS** (pendiente de regresión lineal $SR_{\text{OOS}} \sim SR_{\text{IS}}$).

---

## 9. Pilar 5 — Reparación Crítica de Inferencia en Producción v2.0 → v3.0 (ONNX Real y Paridad Train/Serve)

Esta sección corrige todos los hallazgos **P0** de despliegue (`F-01` y `F-02`):

### 9.1 Exportación Real a ONNX (Adiós `create_synthetic_onnx_model` en Producción)
- En [`src/models/onnx_exporter.py`](file:///c:/Users/ronal/quant-trading-bot/src/models/onnx_exporter.py) y [`src/models/export_champions_to_onnx.py`](file:///c:/Users/ronal/quant-trading-bot/src/models/export_champions_to_onnx.py):
  - **Prohibido** invocar `create_synthetic_onnx_model()` en cualquier ruta de producción (`export_all_champions()`). Mover el helper sintético exclusivamente a `tests/helpers/`.
  - Exportar el **`Pipeline` real completo (`Imputer` + `StandardScaler` + `CalibratedClassifier` / `XGBoost` / `LightGBM` / `LogisticRegression` / `RandomForest`)** a un único grafo `.onnx` usando `skl2onnx` + `onnxmltools` (`convert_xgboost` / `convert_lightgbm`, `options={id(clf): {'zipmap': False}}` para salida tensorial pura `float32[1, 2]`).
  - **Compuerta de Paridad Numérica Obligatoria:** Al exportar cada campeón, evaluar $K = 200$ filas reales del dataset de validación en el modelo nativo y en `ONNXInferenceEngine`:
    $$\max_{k=1,\dots,K} \left|\hat{p}_{\text{native}}(x_k) - \hat{p}_{\text{onnx}}(x_k)\right| < 10^{-5}$$
    Si falla la paridad, **abortar la exportación** y no sobrescribir `onnx_manifest.json`.

### 9.2 Contrato Inmutable `ModelBundle` (`results/campeon_{symbol}.json`)
Unificar entrenamiento e inferencia bajo un contrato versionado `schema_version: "3.0"`:
```json
{
  "schema_version": "3.0",
  "activo": "EURUSD",
  "timeframe": "D1",
  "primary_strategy": {
    "name": "TrendPullbackPrimary",
    "params": {"sma_slow": 200, "ema_fast": 20, "ema_slow": 50, "bilateral": true}
  },
  "cusum_config": {
    "kappa": 1.0,
    "vol_col": "EGARCH_Vol"
  },
  "ffd_d_map": {
    "close": 0.38,
    "DXY_close": 0.42,
    "Yield10Y": 0.35
  },
  "meta_model": {
    "model_type": "XGBOOST_CALIBRATED",
    "onnx_file": "meta_xgboost_eurusd.onnx",
    "sha256": "<digest-sha256>",
    "feature_order": ["close_FFD", "dist_sma200_atr", "EGARCH_Vol", "atr_ratio_14_60", "VIX_close", "DXY_close_FFD", "primary_side", "hmm_prob_choppy"],
    "includes_scaler_in_onnx": true,
    "calibration_method": "sigmoid",
    "brier_score_oos": 0.214,
    "dsr_oos": 0.972,
    "pbo_cscv": 0.041
  },
  "barriers": {
    "k_up": 2.0,
    "k_down": 1.5,
    "max_hold": 10
  },
  "risk": {
    "mc_mdd_95": -0.142,
    "optimal_risk_pct": 0.018,
    "kill_switch_mdd": -0.20,
    "hurdle_multiplier": 2.5
  }
}
```

### 9.3 Correcciones Quirúrgicas en `ExecutionGateway` y `AnalyticsDaemon`
1. **`src/execution/execution_gateway.py`:**
   - Añadir `import json` (corrige `NameError` silencioso).
   - Eliminar el fallback silencioso `prob = 0.50`: si falla la carga o el SHA-256 del ONNX, **fail-closed** (abortar orden + alerta crítica Telegram + registro `MODEL_INTEGRITY_ERROR` en `TradeVault`).
   - Ejecutar `PrimaryStrategyEngine` + `CUSUM` en vivo; si no hay señal primaria (`side == 0`), retornar `NO_PRIMARY_EVENT` sin invocar inferencia ni abrir posición.
   - Aplicar FFD con `ffd_d_map` congelado (sin recalcular `d*` en vivo) y generar el vector exacto en `feature_order` para el grafo ONNX (que ya embebe el `StandardScaler`).
   - Pasar `data_dir=self.results_dir` a `HybridRiskMonitor` para que `InstitutionalRegimeDetector` (HMM 3-estados + KS-Drift) se cargue y ejecute realmente (usando `predict_proba` causal/filtered).
2. **`src/execution/analytics_daemon.py`:**
   - Corregir [`analytics_daemon.py#L85`](file:///c:/Users/ronal/quant-trading-bot/src/execution/analytics_daemon.py#L85): leer las tuplas de 3 features `[net_ret, duration, win_flag]` directamente desde `TradeVault` (`execution_audit_log` / cierres conciliados MT5) en vez de pasar un array 1D `reshape(-1, 1)` que rompe el `StandardScaler` del `StrategyLSTMAutoencoder`.
   - Validar forma `X_recent.shape[1] == autoencoder.scaler.n_features_in_` antes de inferir.

---

## 10. Contratos de Datos, Esquema de Artefactos y Plan de Implementación SDD/TDD

Para ejecutar esta migración sin regresiones y respetando el enfoque **Vertical Slice SDD + TDD**, el trabajo se divide en **5 Fases Secuenciales Verificables** (cada fase exige 100% de tests en verde antes de pasar a la siguiente):

| Fase | Entregables de Código | Pruebas Unitarias Obligatorias (`pytest`) | Criterio de Salida (DoD) |
| :--- | :--- | :--- | :--- |
| **Fase 1: Corrección de Bugs P0 de Integridad y Paridad ONNX** | • Fix `import json`, fail-closed ONNX, `ffd_d_map` congelado y carga HMM en `execution_gateway.py`<br>• Fix 3-feature shape en `analytics_daemon.py`<br>• Exportación real `skl2onnx`/`onnxmltools` en `onnx_exporter.py` y `export_champions_to_onnx.py` | • `test_onnx_real_parity.py`: $\max|\hat{p}_{\text{native}} - \hat{p}_{\text{onnx}}| < 10^{-5}$<br>• `test_gateway_fail_closed.py`: checksum corrupto aborta sin fallback `0.50`<br>• `test_daemon_autoencoder_shape.py`: 3-features exactas | Cero grafos sintéticos en `results/saved_models/`; gateway y daemon 100% funcionales. |
| **Fase 2: Limpieza de Leakage Temporal & Hurst Correcto** | • Purga por `t1 < current_time` en Walk-Forward<br>• `StandardScaler` dentro de `Pipeline` por fold<br>• Eliminar `.bfill()`<br>• Hurst sobre retornos + Variance Ratio en `asset_screener.py` | • `test_no_leakage_strict.py`: inyección de "oracle future shock" en `t+1..t+10` verifica invariancia bit-a-bit de predicciones en `t`<br>• `test_hurst_random_walk.py`: Random Walk simulado arroja $H \in [0.46, 0.54]$ | **0 alertas** en auditoría `timeseries-leakage-guard`. |
| **Fase 3: Capa 0 + Capa 1 + Capa 2 (CUSUM + Modelo Primario + Meta-Labeling Real)** | • `src/preprocessing/cusum_filter.py`<br>• `src/strategies/primary_strategies.py`<br>• Refactor `src/preprocessing/meta_labeling.py` (orientado por `side`, salida `(t0, t1, side, meta_label, net_ret)`)<br>• `src/preprocessing/sample_weights.py` (concurrency, uniqueness, time-decay) | • `test_cusum_filter.py`<br>• `test_primary_meta_labeling.py`: verifica simetría exacta Long vs Short y que `t1 >= t0 + 1`<br>• `test_sample_weights.py`: $\sum w_i = N$ y eventos aislados tienen $u_i = 1.0$ | Dataset de eventos limpio con `t0`, `t1`, `side`, `meta_label` y `sample_weight`. |
| **Fase 4: Entrenamiento Tabular Calibrado + MDA + DSR/PBO Real** | • Features tabulares multi-escala (sin aplanado de 480 cols)<br>• Optuna por `neg_log_loss` + `CalibratedClassifierCV`<br>• `src/evaluation/feature_importance.py` (MDA purgado)<br>• Refactor `cpcv_auditor.py` (CSCV multi-estrategia $T \times N$ + `shift(1)`)<br>• Eliminar OOS threshold snooping y activar DSR gate ($\ge 0.95$) | • `test_probability_calibration.py`: verifica mejora de Brier/ECE<br>• `test_cscv_pbo_formal.py`: estrategia de ruido puro entre 30 trials produce $\text{PBO} > 0.40$; estrategia con señal real produce $\text{PBO} < 0.10$<br>• `test_mda_drops_noise.py`: columna de ruido gaussiano obtiene $\text{MDA} \le 0$ | Campeones promovidos únicamente con `DSR >= 0.95`, `PBO < 0.20`, `ECE < 0.05` y ONNX real verificado. |
| **Fase 5: Opcional Intradía (Information-Driven Bars)** | • `src/preprocessing/information_bars.py`: generador de Tick Bars y Tick Imbalance Bars desde `mt5.copy_ticks_range()` | • `test_information_bars.py`: verifica normalidad mejorada (menor kurtosis / test Jarque-Bera) frente a Time Bars sintéticas con sonrisa de volumen | Habilitado solo si se operan horizontes intradía (`< D1`). |

---

## 11. Matriz de Evolución Completa (v1.0 → v2.0 → v3.0)

| Dimensión | v1.0 (Base) | v2.0 (Infraestructura) | v3.0 (Institucional Completo — Este Spec) |
| :--- | :--- | :--- | :--- |
| **Muestreo & Eventos** | Time Bars D1/H4 cada vela | Igual a v1.0 | **D1 + Filtro CUSUM Simétrico** (y Tick/TIB si intradía) |
| **Generación de Lado (`side`)** | Ninguna (Long-only diario + inversión `1-p` en EURUSD) | Igual a v1.0 | **Modelo Primario Explícito** (Tendencia / Breakout / MR, alto Recall) |
| **Rol del Machine Learning** | Pronóstico direccional diario (`BiLSTM` / `XGBoost`) | Igual a v1.0 | **Meta-Modelo Calibrado (GBDT / Logistic / RF)** como filtro de precisión y sizing |
| **Sample Weights & Purga** | Sin pesos; Walk-Forward con leakage de `max_hold=10` | Igual a v1.0 | **Average Uniqueness + Time Decay + Purga exacta por `[t0, t1]`** |
| **Selección & Anti-Overfitting** | 216 combinaciones sobre OOS sin DSR; PBO sobre 1 modelo | Igual a v1.0 | **Nested CV / umbral analítico de costo + DSR activo + CSCV PBO formal ($T \times N$) + MDA** |
| **Inferencia Producción (ONNX)** | Keras/TF en vivo (~1.4 GB RAM) | Grafo ONNX **sintético** + bugs de `json`/scaler/FFD | **Pipeline Real (`Scaler + Modelo Calibrado`) en ONNX** con paridad `< 1e-5` y `ffd_d_map` congelado |
| **Ejecución, Costos & MLOps** | Monolítico, JSON planos, sin Macro Guard | ✅ **Gateway < 15 ms, SQLite WAL, Cost Gatekeeper, HMM, KS-Drift, Macro Guard** | ✅ **Preservado + HMM/Autoencoder 100% cableados y validados** |

---

> [!IMPORTANT]
> **Regla de Oro para Producción y Entrevistas:**  
> Ningún modelo se despliega con dinero real ni se declara "certificado" hasta completar al menos las **Fases 1 a 4** y confirmar que **los modelos tabulares con Meta-Labeling real + CUSUM + DSR/PBO** superan out-of-sample al benchmark pasivo (`Buy & Hold` y `SMA-200`) **después** de spreads, slippage y swaps reales.
