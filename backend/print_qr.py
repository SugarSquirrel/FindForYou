"""Print a QR code in the terminal for sharing FindForYou.

Usage:
  python print_qr.py --url "http://192.168.0.10:8000/"

Tip:
  - If you expose the server via ngrok, pass the https URL here.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Print an ASCII QR code for a URL")
    parser.add_argument(
        "--url",
        required=True,
        help="Target URL to encode (e.g. http://<ip>:8000/ or https://<ngrok-domain>/)",
    )
    args = parser.parse_args()

    try:
        import qrcode
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'qrcode'. Install with: pip install qrcode[pil]"
        ) from exc

    qr = qrcode.QRCode(border=2)
    qr.add_data(args.url)
    qr.make(fit=True)

    print("\nShare URL:")
    print(args.url)
    print("\nQR Code:\n")
    qr.print_ascii(invert=True)
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
