# Airport Investment Intelligence Agent

Phase 2 repository shell for a one-day Streamlit MVP that screens US airports for capacity-expansion signals.

## Current phase

This phase establishes:

- Python project structure
- validated environment configuration
- dependency declaration
- basic Streamlit entry point
- offline Pytest smoke tests

Airport fixtures, deterministic metrics, scoring, analytical tools, LM Studio routing, and the complete UI are intentionally deferred to later phases.

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

## Tests

```powershell
pytest
```

## Disclaimer

This application screens airport capacity-expansion signals using public aviation data and deterministic analytical proxies. It does not estimate actual investment return.
