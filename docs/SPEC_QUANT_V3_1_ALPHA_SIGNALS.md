# Especificación Quant v3.1 — Señales Ortogonales de Alpha (Fuera de OHLCV Técnico Puro en D1)

> **Propósito:** Definir la hoja de ruta técnica y estadística para pasar de una línea base honesta sin *edge* desplegable (`D1 OHLCV`: `LogLoss ≈ 0.693`, `CSCV PBO = 50%–86%`) a fuentes de señal económicamente fundadas con capacidad realista de superar las compuertas institucionales (`CSCV PBO < 20%`, `DSR >= 0.95`, `OOS LogLoss < 0.67`).  
> **Documento Base:** [`docs/SPEC_QUANT_V3_INSTITUTIONAL_ALPHA.md`](./SPEC_QUANT_V3_INSTITUTIONAL_ALPHA.md)  
> **Fecha:** 26 de Noviembre de 2026

---

## 1. Diagnóstico: Por qué `D1 OHLCV` Técnico Puro No Pasa el Auditor

En la corrida v3.0 sobre `EURUSD`, `Oro` y `SP500`, el pipeline ya estaba libre de *data leakage*, usaba Meta-Labeling real en 2 etapas (`SymmetricCUSUMFilter` + `TrendPullbackPrimary` + `MetaBarrierLabeler`), pesos por unicidad, poda MDA y umbral analítico. Aun así, arrojó:

- `EURUSD`: `CV LogLoss = 0.6871`, `OOS Brier = 0.2511`, `CSCV PBO = 84.29%`
- `Oro`: `CV LogLoss = 0.7054`, `OOS Brier = 0.2473`, `CSCV PBO = 50.00%`
- `SP500`: `CV LogLoss = 0.6929`, `OOS Brier = 0.2502`, `CSCV PBO = 85.71%`

### Tres causas estructurales explican ese resultado
1. **Eficiencia de primer orden en cierres diarios (`D1`):** En activos hiper-líquidos (`EURUSD`, `SP500`, `XAUUSD`), cualquier combinación de `RSI`, `MACD`, `ATR` y medias móviles sobre cierres diarios es pública y arbitrada desde hace décadas. La entropía de la etiqueta binaria queda pegada a $\ln(2) \approx 0.693$.
2. **Bajo tamaño muestral efectivo:** Al aplicar CUSUM + `max_hold = 10` sobre ~15 años de velas diarias, quedan ~600–1,200 eventos (y tras `Sample Uniqueness`, menos de ~350–500 grados de libertad efectivos). Con tan pocas observaciones independientes, ningún árbol o red neuronal logra separar señal débil de ruido sin disparar el `PBO`.
3. **Ausencia de variables de estado exógenas verdaderas:** `VIX_close`, `DXY_close_FFD` y `Yield10Y_FFD` aislados capturan solo una fracción del contexto macro; faltan **diferenciales de tasas**, **estructura temporal de volatilidad**, **spreads de crédito** y **microestructura de flujo de ticks**.

**Conclusión de diseño:** Seguir tuneando hiperparámetros de XGBoost/RandomForest sobre las mismas columnas OHLCV diarias solo aumenta el número de ensayos $N$ y empeora el `DSR`. La solución es cambiar **qué entra al modelo (Capa 0 y Capa 2)** y **qué hipótesis económica propone el lado (Capa 1)**.

---

## 2. Las 4 Fuentes de Señal Ortogonales (Con Hipótesis Económica y Fuentes Reales)

### Fuente 1 — Microestructura Intradía desde Ticks MT5 (`Tick Imbalance Bars` + `VPIN` Proxy)
- **Hipótesis económica:** En ventanas de minutos/horas, el flujo de órdenes informado genera desequilibrios persistentes de presión compradora/vendedora y ampliación asimétrica del spread antes de que el precio diario refleje el movimiento completo.
- **Por qué aplica a MT5 OTC:** Como MT5 devuelve `real_volume == 0` en Forex/CFDs, ya implementamos [`src/preprocessing/information_bars.py`](../src/preprocessing/information_bars.py) con **Tick Bars** y **Tick Imbalance Bars (TIBs)** usando la *Tick Rule* $b_t = \text{sign}(\Delta p_t)$.
- **Implementación concreta:**
  1. Extender `src/data_extractor.py` con `extract_ticks_chunked(symbol, start_dt, end_dt)` usando `mt5.copy_ticks_range()` (en bloques mensuales, persistiendo en Parquet para evitar re-descargas).
  2. Construir barras TIBs y **features microestructurales causales por barra**:
     - `tick_imbalance_norm`: $\frac{\sum_{k \in \text{barra}} b_k}{N_{\text{ticks}}}$ (presión direccional neta).
     - `vpin_proxy`: promedio móvil sobre los últimos $V$ buckets de volumen de ticks de $\frac{|V^+ - V^-|}{V^+ + V^-}$ (toxicidad de flujo / probabilidad de ejecución informada, Easley, López de Prado & O'Hara 2012).
     - `spread_zscore`: $( \text{spread}_{\text{close}} - \mu_{50} ) / \sigma_{50}$ del spread `ask - bid` real registrado en los ticks.
     - `tick_arrival_rate`: ticks por segundo respecto a su mediana móvil de la misma hora UTC (actividad anómala descontando la estacionalidad horaria Londres/NY).
  3. **Filtro horario estricto:** Operar y entrenar solo en sesiones de liquidez institucional (`07:00–17:00 UTC`, apertura Londres a cierre Londres / solapamiento NY), evitando el ruido de spread ancho del rollover (`21:00–23:00 UTC`).

---

### Fuente 2 — Diferenciales de Tasas Reales, Crédito y Estructura Temporal de Volatilidad (Macro Cross-Asset)
En lugar de usar niveles aislados (`Yield10Y`), los activos macro responden a **diferenciales** y **pendientes**:

| Activo | Variables Ortogonales Nuevas | Fuente Gratuita Reproducible | Hipótesis Económica |
| :--- | :--- | :--- | :--- |
| **`EURUSD`** | • `rate_diff_2y = US2Y - DE2Y`<br>• `rate_diff_10y = US10Y - DE10Y`<br>• `delta_diff_5d`, `delta_diff_20d`<br>• `us_curve_slope = US10Y - US2Y` | `FRED` (`DGS2`, `DGS10`, `IRLTLT01DEM156N`) / `yfinance` / `ECB Data Portal API` | El tipo de cambio EUR/USD de mediano plazo sigue el diferencial de tasas reales/nominales de corto plazo (paridad descubierta de tasas y flujos *carry*). |
| **`Oro (XAUUSD)`** | • `us_real_yield_10y` (`DFII10`)<br>• `breakeven_infl_10y` (`T10YIE`)<br>• `delta_real_yield_5d`, `delta_real_yield_20d`<br>• `gold_silver_ratio` (`XAU / XAG`) | `FRED` (`DFII10`, `T10YIE`) + MT5 (`XAGUSD`) | El oro tiene duración real negativa: sube cuando caen las tasas reales TIPS (`DFII10`) o se amplía la inflación esperada sin respuesta de tasas nominales. |
| **`SP500`** | • `vix_term_ratio = VIX / VIX3M`<br>• `vix_contango = VIX3M - VIX`<br>• `hy_credit_spread` (`BAMLH0A0HYM2`)<br>• `delta_hy_spread_5d`<br>• `yield_curve_10y_2y` (`T10Y2Y`) | `yfinance` (`^VIX`, `^VIX3M`) + `FRED` (`BAMLH0A0HYM2`, `T10Y2Y`) | Cuando `VIX / VIX3M > 1.0` (*backwardation* de volatilidad) o se amplía el spread High Yield, el estrés de liquidez precede caídas de renta variable y el *long-only* pierde ventaja. |

> **Regla Anti-Leakage Macro Obligatoria:**  
> - Series de mercado continuo (`^VIX`, `^VIX3M`, `DGS2`, `DGS10`, `DFII10`, `BAMLH0A0HYM2`): aplicar **`shift(1)` estricto** respecto a la fecha de cierre de la barra `t0` del activo operado (el dato del cierre de Nueva York `T-1` es conocido con certeza antes del cierre `T` o apertura `T+1`).  
> - Series con rezago de publicación (ej. datos mensuales/semanales): prohibido alinear por periodo de referencia; usar únicamente `realtime_start` / `observation_date + publication_lag_days` con `.ffill()` exclusivamente hacia adelante (cero `.bfill()`).

---

### Fuente 3 — Nuevas Estrategias Primarias Económicas (Capa 1) y Comparación Sistemática
En v3.0 solo corrimos `TrendPullbackPrimary` por defecto. En v3.1 la Capa 1 debe probar un abanico pequeño y económicamente distinto de reglas causales (sin optimización por gradiente):

1. **`MacroCarryMomentumPrimary` (para `EURUSD` y `Oro`):**
   - `EURUSD`: `side = +1` si `delta_diff_20d(DE2Y - US2Y) > 0` y `Close > EMA(50)`; `-1` en sentido opuesto.
   - `Oro`: `side = +1` si `delta_real_yield_20d < 0` y `Close > SMA(100)`.
2. **`VixRegimeEquityPrimary` (para `SP500`):**
   - `side = +1` solo cuando la estructura de volatilidad está en contango saludable (`VIX / VIX3M < 0.98`) y el spread High Yield no está expandiéndose bruscamente (`delta_hy_spread_5d < +0.15`).
3. **`DonchianBreakoutPrimary` & `VolatilityMeanReversionPrimary`** (ya implementadas en [`src/strategies/primary_strategies.py`](../src/strategies/primary_strategies.py)) — evaluar en paralelo dentro del mismo protocolo cerrado.

---

### Fuente 4 — Efectos de Calendario / Flujo Institucional (Sin Data Mining)
Incorporar únicamente anomalías documentadas con mecanismo económico claro (no días de la semana arbitrarios):
- **`is_turn_of_month`** (último día hábil y primeros 3 días hábiles del mes, donde flujos automáticos de pensiones y rebalanceo de fondos pasivos generan demanda neta en `SP500`).
- **`post_macro_cooldown_flag` + `post_event_direction`:** aprovechar la ventana inmediatamente posterior al cooldown de `MacroHazardGuard` (`+15m` a `+4h` tras FOMC/CPI/NFP en intradía), donde los desbalances de liquidez post-noticia tienden a tener continuidad direccional (*post-announcement drift*).

---

## 3. Protocolo Experimental Cerrado (Anti-Snooping & Presupuesto de Ensayos)

Para no destruir el `DSR` ni volver a sobreajustar probando 500 combinaciones sobre el test set:

1. **Pre-registro de Matriz de Experimentos:** Máximo **3 estrategias primarias × 2 familias tabulares (`Logistic L2`, `XGBoost`/`RF` regularizado)** = **6 ensayos totales** por activo (manteniendo $N_{\text{trials}} = 6$ honesto en la fórmula del DSR).
2. **Validación Anidada / Holdout Intocable:**
   - **70% Desarrollo (`PurgedKFold` 5 folds + poda MDA + calibración `PurgedPlattCalibrator`)**
   - **30% Holdout OOS Final:** se evalúa **una sola vez** por activo al cerrar el diseño.
3. **Compuertas Mínimas de Aceptación para Promoción a Producción (`v3.1`):**
   - `OOS LogLoss < 0.67` (superar holgadamente el azar `0.693`)
   - `OOS ECE < 0.05` (calibración probabilística confiable para `CostSensitiveGatekeeper` y Kelly)
   - `DSR >= 0.95` (calculado con la varianza empírica de los $N=6$ ensayos)
   - `CSCV PBO < 0.20` (matriz $T \times 6$ sobre retornos netos de fricción real)
   - `OOS Net Sharpe > Benchmark (SMA-200 / Buy&Hold)` después de spread, slippage y swap.

---

## 4. Plan de Implementación Modular (Vertical Slices SDD/TDD)

| Slice | Archivos a Crear / Modificar | Pruebas Obligatorias (`pytest`) |
| :--- | :--- | :--- |
| **Slice A — Extractor Macro Real-Time Causal** | • `src/preprocessing/macro_rates.py` (FRED CSV/API pública sin API key obligatoria + `^VIX3M` vía `yfinance`, con cache local en `data/raw/macro/` y `shift(1)` obligatorio). | • `test_macro_rates_causal.py`: verifica ausencia de `.bfill()`, `shift(1)` estricto ante choques futuros en `T`, y cálculo correcto de `rate_diff_2y`, `DFII10` y `VIX / VIX3M`. |
| **Slice B — Ingesta de Ticks MT5 & Features Microestructurales** | • Extender `src/data_extractor.py` con `extract_ticks_chunked()` + guardado Parquet.<br>• Extender `src/preprocessing/information_bars.py` con `compute_tick_microstructure_features()` (`vpin_proxy`, `spread_zscore`, `tick_arrival_rate`, filtro horario `07:00–17:00 UTC`). | • `test_tick_microstructure.py`: verifica causalidad de `vpin_proxy` y `spread_zscore`, filtro horario UTC y paridad en ventana viva vs histórica. |
| **Slice C — Primarias Macro + Meta-Features v3.1** | • Añadir `MacroCarryMomentumPrimary` y `VixRegimeEquityPrimary` en `src/strategies/primary_strategies.py`.<br>• Extender `src/preprocessing/tabular_features.py` con diferenciales de tasas, `vix_term_ratio`, `vpin_proxy` e `is_turn_of_month`. | • `test_primary_macro_strategies.py`: test *Oracle Future Shock* garantizando que las nuevas features y primarias no miran `t+1`. |
| **Slice D — Benchmark Comparativo v3.0 vs v3.1** | • Extender `src/main_training_v3.py` con `--mode {d1_ohlcv, d1_macro, intraday_tib}` y tabla comparativa pre-registrada (`N_trials = 6`). | • Corrida real reproducible sobre `EURUSD`, `Oro`, `SP500` registrando artefactos y métricas exactas en `README.md` y MLflow. |
