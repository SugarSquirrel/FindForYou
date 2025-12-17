"""Print a QR code in the terminal for sharing FindForYou.

Usage:
  python print_qr.py --url "http://192.168.0.10:8000/"

Tip:
  - If you expose the server via ngrok, pass the https URL here.
"""

from __future__ import annotations

import argparse
import json
from urllib.request import urlopen
from urllib.error import URLError


def _get_ngrok_public_url() -> str | None:
    try:
        with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=0.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError, OSError):
        return None

    tunnels = payload.get("tunnels") or []
    if not isinstance(tunnels, list):
        return None

    for tunnel in tunnels:
        public_url = tunnel.get("public_url") if isinstance(tunnel, dict) else None
        if isinstance(public_url, str) and public_url.startswith("https://"):
            return public_url.rstrip("/") + "/"
    for tunnel in tunnels:
        public_url = tunnel.get("public_url") if isinstance(tunnel, dict) else None
        if isinstance(public_url, str) and public_url.startswith("http://"):
            return public_url.rstrip("/") + "/"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Print an ASCII QR code for a URL")
    parser.add_argument(
        "--url",
        required=False,
        help="Target URL to encode (e.g. http://<ip>:8000/ or https://<ngrok-domain>/)",
    )
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Auto-detect ngrok public URL from http://127.0.0.1:4040/api/tunnels",
    )
    args = parser.parse_args()

    url = args.url
    if args.ngrok:
        url = _get_ngrok_public_url()
        if not url:
            raise SystemExit("Cannot detect ngrok URL. Is ngrok running with the local web UI (4040)?")

    if not url:
        raise SystemExit("Please provide --url or use --ngrok")

    try:
        import qrcode
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'qrcode'. Install with: pip install qrcode[pil]"
        ) from exc

    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)

    print("\nShare URL:")
    print(url)
    print("\nQR Code:\n")
    qr.print_ascii(invert=True)
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
