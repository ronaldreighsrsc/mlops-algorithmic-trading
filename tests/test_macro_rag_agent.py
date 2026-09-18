"""
Tests unitarios para el Guardián Macro Pre-News y Reportes Forenses RCA.
"""

from datetime import datetime, timezone, timedelta
import tempfile
import os
import pytest
from src.macro.macro_rag_agent import MacroHazardGuard
from src.macro.economic_calendar import EconomicCalendarProvider


@pytest.fixture
def macro_guard():
    return MacroHazardGuard(pre_event_freeze_minutes=30, post_event_cooldown_minutes=15)


def test_market_safe_when_no_events(macro_guard):
    """Sin eventos programados el mercado debe reportarse como SAFE."""
    is_safe, reason, event = macro_guard.is_market_safe(
        symbol="EURUSD",
        scheduled_events=[],
    )
    assert is_safe is True
    assert "SAFE" in reason
    assert event is None


def test_freeze_pre_news_window(macro_guard):
    """Un evento de alto impacto (FOMC) 15 minutos en el futuro debe activar FREEZE_PRE_NEWS."""
    now = datetime(2026, 10, 15, 17, 45, 0, tzinfo=timezone.utc)
    event_time = datetime(2026, 10, 15, 18, 0, 0, tzinfo=timezone.utc)  # En 15 min

    events = [{
        "title": "FOMC Interest Rate Decision",
        "currency": "USD",
        "impact": "HIGH",
        "time_utc": event_time,
    }]

    is_safe, reason, event = macro_guard.is_market_safe(
        symbol="EURUSD",
        scheduled_events=events,
        current_time_utc=now,
    )

    assert is_safe is False
    assert "FREEZE_PRE_NEWS" in reason
    assert event is not None
    assert event["title"] == "FOMC Interest Rate Decision"


def test_cooldown_post_news_window(macro_guard):
    """Un evento publicado hace 8 minutos (CPI) debe mantener COOLDOWN_POST_NEWS activo."""
    now = datetime(2026, 10, 15, 18, 8, 0, tzinfo=timezone.utc)
    event_time = datetime(2026, 10, 15, 18, 0, 0, tzinfo=timezone.utc)  # Hace 8 min

    events = [{
        "title": "US CPI m/m",
        "currency": "USD",
        "impact": "HIGH",
        "time_utc": event_time,
    }]

    is_safe, reason, event = macro_guard.is_market_safe(
        symbol="SP500",
        scheduled_events=events,
        current_time_utc=now,
    )

    assert is_safe is False
    assert "COOLDOWN_POST_NEWS" in reason
    assert event is not None


def test_unrelated_currency_does_not_freeze(macro_guard):
    """Un evento en GBP o JPY no debe congelar EURUSD si no tiene relación directa."""
    now = datetime(2026, 10, 15, 17, 45, 0, tzinfo=timezone.utc)
    event_time = datetime(2026, 10, 15, 18, 0, 0, tzinfo=timezone.utc)

    events = [{
        "title": "BOJ Monetary Policy Statement",
        "currency": "JPY",
        "impact": "HIGH",
        "time_utc": event_time,
    }]

    is_safe, reason, event = macro_guard.is_market_safe(
        symbol="EURUSD",
        scheduled_events=events,
        current_time_utc=now,
    )

    assert is_safe is True
    assert "SAFE" in reason


def test_economic_calendar_provider_flow():
    """Verifica la adición, filtrado y persistencia del calendario económico."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_cache = f.name

    try:
        provider = EconomicCalendarProvider(cache_file=temp_cache)
        t_event = datetime.now(timezone.utc) + timedelta(hours=2)

        provider.add_event(
            title="Non-Farm Payrolls",
            currency="USD",
            time_utc=t_event,
            impact="HIGH",
            forecast="180K",
            previous="175K"
        )
        provider.save_cache()

        # Recargar desde caché
        reloaded = EconomicCalendarProvider(cache_file=temp_cache)
        events = reloaded.get_upcoming_events(window_hours=6, currency="USD")
        assert len(events) == 1
        assert events[0]["title"] == "Non-Farm Payrolls"
    finally:
        if os.path.exists(temp_cache):
            os.remove(temp_cache)


def test_generate_forensic_rca(macro_guard):
    """Verifica la generación de la bitácora RCA forense con correlación de noticias."""
    exit_time = datetime(2026, 10, 15, 18, 5, 0, tzinfo=timezone.utc)
    trade_event = {
        "ticket_id": 987654,
        "exit_time": exit_time,
        "pnl_usd": -125.50,
        "exit_reason": "STOP_LOSS",
    }
    recent_events = [{
        "title": "Fed Interest Rate Decision",
        "currency": "USD",
        "impact": "HIGH",
        "time_utc": datetime(2026, 10, 15, 18, 0, 0, tzinfo=timezone.utc),
        "forecast": "5.25%",
        "previous": "5.50%",
    }]

    report = macro_guard.generate_forensic_rca("EURUSD", trade_event, recent_events)
    assert "Bitácora Forense Post-Trade (RCA) - Ticket #987654" in report
    assert "Fed Interest Rate Decision" in report
    assert "**Diagnóstico:** La salida estuvo fuertemente influenciada por volatilidad macroexógena." in report
