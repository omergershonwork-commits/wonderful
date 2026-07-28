"""Real Streamlit integration smoke test; requires declared project dependencies."""
from streamlit.testing.v1 import AppTest
from src.config import get_settings

def test_streamlit_app_starts(monkeypatch):
    monkeypatch.setenv('USE_LLM','false')
    monkeypatch.setenv('HTTP_TIMEOUT_SECONDS','1')
    get_settings.cache_clear()
    app=AppTest.from_file('app.py').run(timeout=15)
    assert not app.exception
    assert app.title[0].value=='Airport Investment Intelligence Agent'

def test_follow_up_input_remains_after_suggested_prompt(monkeypatch):
    monkeypatch.setenv("USE_LLM", "false")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    app = AppTest.from_file("app.py").run(timeout=15)

    assert len(app.chat_input) == 1
    app.button(key="prompt-3").click().run(timeout=15)

    assert not app.exception
    assert len(app.chat_input) == 1
    assert app.chat_input(key="follow-up-input") is not None
    assert len(app.chat_message) >= 2

