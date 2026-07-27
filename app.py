"""Streamlit entry point for the Airport Investment Intelligence Agent."""

import streamlit as st

from src.config import get_settings


DISCLAIMER = (
    "This application screens airport capacity-expansion signals using public "
    "aviation data and deterministic analytical proxies. It does not estimate "
    "actual investment return."
)


def render_app() -> None:
    """Render the Phase 2 application shell."""

    settings = get_settings()

    st.set_page_config(
        page_title="Airport Investment Intelligence Agent",
        page_icon="✈️",
        layout="wide",
    )

    st.title("Airport Investment Intelligence Agent")
    st.caption("One-day MVP repository shell")

    with st.sidebar:
        st.header("Runtime configuration")
        st.write(f"LLM enabled: **{settings.use_llm}**")
        st.write(f"LM Studio endpoint: `{settings.llm_base_url}`")
        st.write(f"Configured model: `{settings.llm_model or 'not set'}`")
        st.write(f"Live public data enabled: **{settings.use_live_data}**")

    st.info(
        "Phase 2 is ready. Deterministic airport data, metrics, tools, and chat "
        "routing will be added in subsequent phases."
    )

    st.subheader("Planned analyst questions")
    st.markdown(
        """
- Which airports in New England are strong candidates for terminal expansion?
- Compare LAX and Santa Ana airport congestion levels.
- What percentage of flights from Anchorage are long haul?
- What is the unmet flight demand at SFO and why?
"""
    )

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    render_app()
