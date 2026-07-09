"""Nickplots (HTML/CSS front via PyWebView, Python/matplotlib backend).
Run: python main_web.py
"""
import os
import sys
import webview
from api import Api


def resource_html():
    """Find index.html in dev or packaged builds, even if the web/ folder was
    flattened on download (index.html sitting next to the script)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    for rel in ("web/index.html", "index.html"):
        p = os.path.normpath(os.path.join(base, rel))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "index.html not found. Expected at '<folder>/web/index.html' "
        "(or next to main_web.py). Make sure the 'web' folder was included.")


def main():
    api = Api()
    window = webview.create_window(
        "Nickplots", resource_html(), js_api=api,
        width=1380, height=880, min_size=(1120, 720), background_color="#0e1015")
    api._window = window
    webview.start()   # uses the OS native WebView (WebView2/WebKit/WebKitGTK)


if __name__ == "__main__":
    main()
