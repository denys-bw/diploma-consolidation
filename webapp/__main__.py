from __future__ import annotations

import argparse
import threading
import webbrowser

from webapp import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
BROWSER_DELAY_SECONDS = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m webapp", description="Веб-застосунок консолідації даних")
    parser.add_argument("--host", default=DEFAULT_HOST, help="адреса, на якій працює застосунок")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="порт застосунку")
    parser.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    arguments = parser.parse_args()
    app = create_app()
    url = f"http://{arguments.host}:{arguments.port}/"
    print(f"Застосунок працює на {url} — щоб зупинити, натисніть Ctrl+C")
    if not arguments.no_browser:
        threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, args=(url,)).start()
    app.run(host=arguments.host, port=arguments.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
