"""
Proveedor de Calendario Económico Macroeconómico.
Gestiona la ingesta de eventos fundamentales de alto impacto (FOMC, CPI, NFP, Tasas de Interés)
con soporte para caché local offline y proveedores en línea.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("EconomicCalendar")


class EconomicCalendarProvider:
    """
    Proveedor de eventos macroeconómicos programados.
    """

    def __init__(self, cache_file: Optional[str] = None):
        if cache_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_file = os.path.join(base_dir, "cache", "economic_calendar.json")
        self.cache_file = cache_file
        self.events: List[Dict] = []
        self._load_cache()

    def _load_cache(self):
        """Carga eventos guardados en caché si existen."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.events = []
                    for ev in data:
                        # Parsear datetime ISO
                        t = datetime.fromisoformat(ev["time_utc"])
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=timezone.utc)
                        ev["time_utc"] = t
                        self.events.append(ev)
            except Exception as e:
                logger.warning(f"No se pudo cargar el calendario desde caché: {e}")

    def save_cache(self):
        """Persiste los eventos en disco."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        serializable = []
        for ev in self.events:
            d = dict(ev)
            if isinstance(d["time_utc"], datetime):
                d["time_utc"] = d["time_utc"].isoformat()
            serializable.append(d)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def add_event(
        self,
        title: str,
        currency: str,
        time_utc: datetime,
        impact: str = "HIGH",
        forecast: Optional[str] = None,
        previous: Optional[str] = None,
    ):
        """Agrega un evento al calendario programado."""
        if time_utc.tzinfo is None:
            time_utc = time_utc.replace(tzinfo=timezone.utc)

        ev = {
            "title": title,
            "currency": currency.upper(),
            "time_utc": time_utc,
            "impact": impact.upper(),
            "forecast": forecast,
            "previous": previous,
        }
        self.events.append(ev)

    def get_upcoming_events(
        self,
        now_utc: Optional[datetime] = None,
        window_hours: float = 24.0,
        currency: Optional[str] = None,
        impact_level: Optional[str] = "HIGH",
    ) -> List[Dict]:
        """
        Retorna los eventos programados dentro de la ventana especificada.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        end_window = now_utc + timedelta(hours=window_hours)
        filtered = []

        for ev in self.events:
            ev_time = ev["time_utc"]
            if now_utc - timedelta(minutes=30) <= ev_time <= end_window:
                if currency and currency.upper() not in ev["currency"].upper():
                    continue
                if impact_level and ev["impact"] != impact_level.upper():
                    continue
                filtered.append(ev)

        return sorted(filtered, key=lambda x: x["time_utc"])
