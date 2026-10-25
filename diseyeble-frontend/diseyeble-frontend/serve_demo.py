"""Optional: serves this demo page on http://localhost:8000 and lets its "Start engine" button run the engine (real computer control).
Plain Allow camera needs none of this: the page tracks your face by itself.

Run it from any Python 3 (no packages needed):
    python serve_demo.py --engine-dir C:\\path\\to\\diseyeble
Then open http://localhost:8000/#demo and click Start engine. No window opens; the page shows Engine live once it is up. Ctrl+C here stops this server and the engine it started.

The page cannot start programs by itself, so this small local server does it. It only listens on 127.0.0.1,
only ever runs the engine command given here, and refuses requests that don't come from this page.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent
ENGINE_PORT = 8765
LOG_PATH = Path(tempfile.gettempdir()) / "diseyeble-engine.log"


def engine_listening(port: int = ENGINE_PORT) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_engine_dir() -> Path | None:
    """A folder named diseyeble next to this page's folders (the usual worktree layout), then any folder above."""
    levels = [PAGE_DIR, *PAGE_DIR.parents]
    for level in levels:
        if (level / "diseyeble" / "engine" / "main.py").exists():
            return level / "diseyeble"
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
        # No console window: the page shows what the engine does. Its own printout goes to a log file.
        self.log = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        self.proc = subprocess.Popen([self.python, "-m", self.module], cwd=self.dir, stdout=self.log,
                                     stderr=subprocess.STDOUT, **kwargs)
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
            if self.headers.get("X-Diseyeble") != "1" or self.headers.get("Origin") not in origins:
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
    args = parser.parse_args()

    engine_dir = args.engine_dir or find_engine_dir()
    if not engine_dir or not (engine_dir / "engine").is_dir():
        sys.exit("Give the folder that holds engine/main.py:  python serve_demo.py --engine-dir C:\\path\\to\\diseyeble")
    engine = Engine(engine_dir.resolve(), args.python, args.module)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.port))
    print(f"Demo page: http://localhost:{args.port}/#demo   engine folder: {engine.dir}")
    print("Click \"Start engine\" on the page to start the engine (no window; its log is " + str(LOG_PATH) + "). Ctrl+C stops everything.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
