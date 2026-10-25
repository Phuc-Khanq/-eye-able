"""Serves this demo page on http://localhost:8000 and starts the engine when the page's Allow camera is clicked.

Run it from any Python 3 (no packages needed):
    python serve_demo.py --engine-dir C:\\path\\to\\eyeable
Then open http://localhost:8000/#demo. The engine's own window (python -m engine.main) opens when you click
Allow camera, and the page shows Engine live once it is up. Ctrl+C here stops this server and the engine it started.

The page cannot start programs by itself, so this small local server does it. It only listens on 127.0.0.1,
only ever runs the engine command given here, and refuses requests that don't come from this page.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent
ENGINE_PORT = 8765


def engine_listening(port: int = ENGINE_PORT) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_engine_dir() -> Path | None:
    """A folder named eyeable next to this page's folders (the usual worktree layout), then any folder above."""
    levels = [PAGE_DIR, *PAGE_DIR.parents]
    for level in levels:
        if (level / "eyeable" / "engine" / "main.py").exists():
            return level / "eyeable"
    for level in levels:
        if (level / "engine" / "main.py").exists():
            return level
    return None


class Engine:
    def __init__(self, engine_dir: Path, python: str | None, module: str) -> None:
        self.dir = engine_dir
        self.module = module
        venv = engine_dir / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
        self.python = python or (str(venv) if venv.exists() else sys.executable)
        self.proc: subprocess.Popen | None = None

    def start(self) -> str:
        if engine_listening() or (self.proc and self.proc.poll() is None):
            return "already running"
        kwargs = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if sys.platform == "win32" else {}
        self.proc = subprocess.Popen([self.python, "-m", self.module], cwd=self.dir, **kwargs)
        return "started"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def make_handler(engine: Engine, port: int):
    origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/status":
                return self._json(200, {"launcher": True, "engine": engine_listening()})
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/start-engine":
                return self._json(404, {"ok": False, "error": "not found"})
            # a custom header plus this page's own origin: another website can't make us start anything
            if self.headers.get("X-Eyeable") != "1" or self.headers.get("Origin") not in origins:
                return self._json(403, {"ok": False, "error": "not from the demo page"})
            try:
                return self._json(200, {"ok": True, "result": engine.start()})
            except OSError as exc:
                return self._json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt, *args):
            if "/api/" in self.path:
                print("[launcher]", fmt % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-dir", type=Path, help="the folder that holds engine/main.py")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python", help="Python that runs the engine (default: the .venv in the engine folder)")
    parser.add_argument("--module", default="engine.main", help="what to run with -m (for tests)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the page in your browser")
    args = parser.parse_args()

    engine_dir = args.engine_dir or find_engine_dir()
    if not engine_dir or not (engine_dir / "engine").is_dir():
        sys.exit("Give the folder that holds engine/main.py:  python serve_demo.py --engine-dir C:\\path\\to\\eyeable")
    engine = Engine(engine_dir.resolve(), args.python, args.module)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.port))
    url = f"http://localhost:{args.port}/#demo"
    print(f"Demo page: {url}   engine folder: {engine.dir}")
    print("Click Allow camera on the page to start the engine. Ctrl+C stops everything.")
    if args.port != 8000:
        print("Note: the engine only talks to pages from localhost:8000 (settings.ws_allowed_origins); add this port there.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
