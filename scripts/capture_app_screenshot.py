"""Capture a real Streamlit demonstration screenshot for project documentation."""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8501")
OUTPUT_PATH = Path(
    os.environ.get("SCREENSHOT_PATH", "artifacts/airport-intelligence-demo.png")
)
PROMPT = "Which airports in New England are strong candidates for terminal expansion?"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.goto(APP_URL, wait_until="networkidle", timeout=60_000)
        page.get_by_role("heading", name="Airport Investment Intelligence Agent").wait_for(
            timeout=30_000
        )

        prompt_button = page.get_by_role("button", name=PROMPT)
        prompt_button.click()
        page.get_by_role(
            "heading", name="Ranked airport expansion candidates"
        ).wait_for(timeout=60_000)

        page.screenshot(path=str(OUTPUT_PATH), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
