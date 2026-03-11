"""
run_app.py — The ON switch.

This is the only file you run. It starts the server.
Everything else gets pulled in automatically.

Usage:
    python run_app.py                   # localhost:5000
    python run_app.py --port 8080       # custom port
    python run_app.py --host 0.0.0.0    # expose on local network
    python run_app.py --debug           # auto-reload on code changes
"""

import argparse
from app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Segmentation Server")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    flask_app = create_app()

    print("\n" + "=" * 54)
    print("  Water Segmentation — Flask Server")
    print(f"  URL   → http://{args.host}:{args.port}")
    print(f"  Debug → {args.debug}")
    print("  Ctrl+C to stop")
    print("=" * 54 + "\n")

    flask_app.run(
        host  = args.host,
        port  = args.port,
        debug = args.debug
    )