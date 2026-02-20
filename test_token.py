#!/usr/bin/env python3
"""
test_chromium.py - Verify Playwright and Chromium work
"""
from playwright.sync_api import sync_playwright


def main():
    print("Testing Playwright + Chromium...")

    with sync_playwright() as p:
        print("✓ Playwright started")

        # Try to launch browser
        browser = p.chromium.launch(headless=True)
        print("✓ Chromium launched")

        # Create page
        page = browser.new_page()
        print("✓ Page created")

        # Navigate to example.com
        page.goto("https://example.com")
        print(f"✓ Navigated to: {page.url}")
        print(f"✓ Page title: {page.title()}")

        browser.close()
        print("✓ Browser closed")
        print("\nAll tests passed! Playwright is working.")


if __name__ == "__main__":
    main()