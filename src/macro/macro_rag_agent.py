"""
Agente Macro-Forense y Guardián Pre-News Institucional (v2.0).
Monitorea eventos de alto impacto económico (FOMC, CPI, NFP, Tasas) para:
1. Congelar la apertura de órdenes de forma preventiva 30 min antes del evento.
2. Mantener un periodo de enfriamiento (cooldown) de 15 min después del evento.
3. Generar reportes forenses RCA (Root Cause Analysis) post-trade para auditorías institucionales y Darwinex.
"""

from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("MacroHazardGuard")


class MacroHazardGuard:
    """
    Guardián Pre-News y Analista Forense Post-Trade.
    """

    HIGH_IMPACT_KEYWORDS = [
        "FOMC",
        "INTEREST RATE",
        "FED",
        "CPI",
        "NON-FARM",
        "NFP",
        "GDP",
        "CENTRAL BANK",
        "TPM",
        "INFLATION",
        "ECB",
        "UNEMPLOYMENT",
        "PAYROLLS",
    ]

    def __init__(
        self,
        pre_event_freeze_minutes: int = 30,
        post_event_cooldown_minutes: int = 15,
    ):
        self.pre_freeze = timedelta(minutes=pre_event_freeze_minutes)
        self.post_cooldown = timedelta(minutes=post_event_cooldown_minutes)

    def _symbol_matches_currency(self, symbol: str, currency: str) -> bool:
        """Determina si la divisa del evento macro tiene impacto sobre el activo operado."""
        sym_clean = symbol.upper().replace("/", "").replace("_", "")
        curr = currency.upper()

        # Mapeos de activos cruzados comunes
        if curr == "USD":
            # Afecta a majors FX, Oro y SP500
            return any(k in sym_clean for k in ["USD", "ORO", "GOLD", "XAU", "SP500", "US500", "SPX"])
        if curr == "EUR":
            return any(k in sym_clean for k in ["EUR"])
        if curr == "CLP":
            return any(k in sym_clean for k in ["CLP", "ECH"])

        return curr in sym_clean

    def is_market_safe(
        self,
        symbol: str,
        scheduled_events: List[Dict],
        current_time_utc: Optional[datetime] = None,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Verifica si existen eventos de alto impacto inmediatos para el activo.
        Retorna: (is_safe, reason, triggering_event)
        """
        now = current_time_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        for event in scheduled_events:
            title = event.get("title", "").upper()
            currency = event.get("currency", "").upper()
            impact = event.get("impact", "").upper()
            event_time = event.get("time_utc")

            if event_time is None:
                continue

            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            # Verificar relevancia del activo
            if self._symbol_matches_currency(symbol, currency):
                is_high_impact = (impact == "HIGH") or any(kw in title for kw in self.HIGH_IMPACT_KEYWORDS)

                if is_high_impact:
                    time_diff = event_time - now

                    # 1. Ventana Pre-Evento (Congelamiento)
                    if timedelta(seconds=0) <= time_diff <= self.pre_freeze:
                        mins_left = int(time_diff.total_seconds() / 60)
                        reason = (
                            f"FREEZE_PRE_NEWS: Evento crítico macro '{title}' ({currency}) "
                            f"programado en {mins_left} minutos. Apertura de órdenes congelada."
                        )
                        return False, reason, event

                    # 2. Ventana Post-Evento (Enfriamiento / Cooldown)
                    time_since = now - event_time
                    if timedelta(seconds=0) <= time_since <= self.post_cooldown:
                        mins_ago = int(time_since.total_seconds() / 60)
                        reason = (
                            f"COOLDOWN_POST_NEWS: Evento crítico macro '{title}' ({currency}) "
                            f"publicado hace {mins_ago} minutos. Mercado en fase de estabilización de spread."
                        )
                        return False, reason, event

        return True, "SAFE: Sin noticias macroeconómicas de alto riesgo inmediatas.", None

    def generate_forensic_rca(
        self,
        symbol: str,
        trade_event: Dict,
        recent_events: List[Dict],
    ) -> str:
        """
        Genera un informe institucional Root Cause Analysis (RCA)
        cruzando el resultado de una operación (ej. Stop Loss o salida inesperada)
        con el contexto macroeconómico reciente.
        """
        ticket = trade_event.get("ticket_id", "N/A")
        exit_time = trade_event.get("exit_time", datetime.now(timezone.utc).isoformat())
        pnl = trade_event.get("pnl_usd", 0.0)
        reason = trade_event.get("exit_reason", "STOP_LOSS")

        # Buscar si hubo noticias cercanas al momento del cierre (ventana +/- 2 horas)
        if isinstance(exit_time, str):
            t_exit = datetime.fromisoformat(exit_time)
        else:
            t_exit = exit_time
        if t_exit.tzinfo is None:
            t_exit = t_exit.replace(tzinfo=timezone.utc)

        correlated_news = []
        for ev in recent_events:
            ev_t = ev.get("time_utc")
            if isinstance(ev_t, str):
                ev_t = datetime.fromisoformat(ev_t)
            if ev_t.tzinfo is None:
                ev_t = ev_t.replace(tzinfo=timezone.utc)

            delta = abs((ev_t - t_exit).total_seconds()) / 60.0
            if delta <= 120.0 and self._symbol_matches_currency(symbol, ev.get("currency", "")):
                correlated_news.append((ev, int(delta)))

        rca_lines = [
            f"# 📋 Bitácora Forense Post-Trade (RCA) - Ticket #{ticket}",
            f"- **Activo:** {symbol}",
            f"- **Fecha/Hora Cierre:** {t_exit.isoformat()}",
            f"- **Motivo de Salida:** {reason}",
            f"- **Resultado Financiero:** ${pnl:.2f} USD",
            "",
            "## Correlación Macroeconómica:",
        ]

        if correlated_news:
            rca_lines.append("Se identificaron eventos fundamentales correlacionados temporalmente:")
            for ev, delta_m in correlated_news:
                rca_lines.append(
                    f"  - **{ev.get('title')}** ({ev.get('currency')}) - Impacto: {ev.get('impact')} | "
                    f"Diferencia: {delta_m} minutos respecto al cierre. (Pronóstico: {ev.get('forecast')}, Previo: {ev.get('previous')})"
                )
            rca_lines.append("\n**Diagnóstico:** La salida estuvo fuertemente influenciada por volatilidad macroexógena.")
        else:
            rca_lines.append("No se detectaron noticias macroeconómicas de alto impacto en una ventana de +/- 2 horas.")
            rca_lines.append("\n**Diagnóstico:** Movimiento endógeno puramente microestructural de mercado.")

        return "\n".join(rca_lines)
