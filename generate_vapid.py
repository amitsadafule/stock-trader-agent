"""
Generate VAPID keys for Web Push notifications.
Run ONCE:  python generate_vapid.py

Creates:
  vapid_private.pem  — keep secret, never share
  vapid_public.txt   — sent to browser for push subscription
"""
from pathlib import Path

try:
    from py_vapid import Vapid
except ImportError:
    try:
        from pywebpush import Vapid
    except ImportError:
        print("Installing py-vapid...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "py-vapid"])
        from py_vapid import Vapid

PRIVATE_KEY = "vapid_private.pem"
PUBLIC_KEY  = "vapid_public.txt"

if Path(PRIVATE_KEY).exists():
    print(f"✅ VAPID keys already exist.")
    print(f"   Private: {PRIVATE_KEY}")
    print(f"   Public:  {PUBLIC_KEY}")
    print("\nPublic key (paste into browser if needed):")
    print(Path(PUBLIC_KEY).read_text().strip())
else:
    v = Vapid()
    v.generate_keys()
    v.save_key(PRIVATE_KEY)
    pub = v.public_key.public_bytes(
        __import__('cryptography').hazmat.primitives.serialization.Encoding.X962,
        __import__('cryptography').hazmat.primitives.serialization.PublicFormat.UncompressedPoint
    )
    import base64
    pub_b64 = base64.urlsafe_b64encode(pub).rstrip(b"=").decode("utf-8")
    Path(PUBLIC_KEY).write_text(pub_b64)
    print("✅ VAPID keys generated!")
    print(f"   Private key saved to: {PRIVATE_KEY}")
    print(f"   Public key saved to:  {PUBLIC_KEY}")
    print(f"\n   Public key: {pub_b64}")

print("\n▶ Now run:  python app.py")
