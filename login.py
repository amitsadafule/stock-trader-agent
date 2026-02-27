"""
First-time login script. Run this once to generate and save your access token.
After this, agent.py will auto-load the token daily.
Note: Zerodha tokens expire daily — you'll need to re-run this once each morning.
"""
from kiteconnect import KiteConnect
from config import ZERODHA_CONFIG, PATHS
from pathlib import Path

def login():
    kite = KiteConnect(api_key=ZERODHA_CONFIG["api_key"])
    print("\n" + "="*60)
    print("ZERODHA FIRST-TIME LOGIN")
    print("="*60)
    print(f"\n👉 Open this URL in your browser:\n\n   {kite.login_url()}\n")
    print("After logging in, you'll be redirected to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXXX&status=success\n")
    token = input("Paste the request_token from that URL: ").strip()
    data = kite.generate_session(token, api_secret=ZERODHA_CONFIG["api_secret"])
    access_token = data["access_token"]
    Path(PATHS["token_file"]).write_text(access_token)
    print(f"\n✅ Access token saved to {PATHS['token_file']}")
    print("▶  Now run:  python agent.py")

if __name__ == "__main__":
    login()
