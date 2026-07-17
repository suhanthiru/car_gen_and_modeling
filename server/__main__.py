"""Run the server: `python -m server`

Prints the LAN URL to open on a phone. Devices must be on the same Wi-Fi; the
server is not reachable from the internet (see README for the Tailscale path
when you want off-network access).
"""
from __future__ import annotations

import socket

import uvicorn

from server.config import CONFIG


def lan_ip() -> str:
    """This machine's address on the local network."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # no packets are sent; this just asks the OS which interface would route out
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    url = f"http://{lan_ip()}:{CONFIG.port}"
    print("\n" + "=" * 62)
    print("  cargen server")
    print("=" * 62)
    print(f"  Capture (open this on your phone) : {url}/")
    print(f"  Viewer                            : {url}/viewer/")
    print(f"  Vehicles are saved to             : {CONFIG.storage_root}")
    print(f"  auto-merge                        : "
          f"{'ON' if CONFIG.auto_merge else 'OFF (duplicates need approval)'}")
    print("=" * 62 + "\n")
    uvicorn.run("server.app:app", host=CONFIG.host, port=CONFIG.port, log_level="info")


if __name__ == "__main__":
    main()
