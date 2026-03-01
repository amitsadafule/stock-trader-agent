#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# Zerodha Trading Agent — One-Click Android / Termux Setup
# ============================================================
# Run once inside Termux:
#   bash setup_android.sh
# ============================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

step()  { echo -e "\n${BOLD}${BLUE}━━ $1 ${NC}"; }
ok()    { echo -e "${GREEN}  ✓ $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
info()  { echo -e "  $1"; }

# ── Banner ────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔════════════════════════════════════════════╗"
echo "  ║   Zerodha Trading Agent — Android Setup   ║"
echo "  ╚════════════════════════════════════════════╝"
echo -e "${NC}"

REPO_DIR="$HOME/stock-trader-agent"

# ── Step 1: Termux packages ───────────────────────────────
step "1/6  Update Termux & install packages"
pkg update -y && pkg upgrade -y
pkg install -y python git tmux openssh
ok "python, git, tmux installed"

# ── Step 2: Get project files ─────────────────────────────
step "2/6  Project files"

if [ -f "app.py" ] && [ -f "requirements.txt" ]; then
    # Running from inside the project directory already
    REPO_DIR="$(pwd)"
    ok "Already in project directory: $REPO_DIR"
elif [ -d "$REPO_DIR/.git" ]; then
    warn "Found existing repo at $REPO_DIR — pulling latest"
    git -C "$REPO_DIR" pull
else
    echo ""
    echo -e "  Choose how to get the project files:"
    echo -e "  ${BOLD}[1]${NC} Clone from GitHub (if repo is public/private)"
    echo -e "  ${BOLD}[2]${NC} Files already copied manually to $REPO_DIR"
    echo ""
    read -rp "  Enter choice [1/2]: " CHOICE

    if [ "$CHOICE" = "1" ]; then
        read -rp "  GitHub repo URL: " REPO_URL
        git clone "$REPO_URL" "$REPO_DIR"
        ok "Cloned to $REPO_DIR"
    else
        warn "Make sure files are in $REPO_DIR before continuing"
        mkdir -p "$REPO_DIR"
    fi
fi

cd "$REPO_DIR"

# Verify essential files exist
for f in app.py agent.py config.py requirements.txt generate_vapid.py; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}  ✗ Missing file: $f — copy all project files to $REPO_DIR first${NC}"
        exit 1
    fi
done
ok "All project files present"

# ── Step 3: Python dependencies ───────────────────────────
step "3/6  Install Python packages"
# pandas dropped — pure Python EMA, no compilation needed.
# cryptography (pulled by pywebpush) ships as a pre-built pkg — avoids Rust compilation.
# `pip install --upgrade pip` is intentionally blocked by Termux (termux#3235).
pkg install -y python-cryptography
pip install -r requirements.txt
ok "All packages installed"

# ── Step 4: VAPID keys (push notifications) ───────────────
step "4/6  Push notification keys (VAPID)"
if [ -f vapid_private.pem ] && [ -f vapid_public.txt ]; then
    warn "Keys already exist — skipping generation"
else
    python generate_vapid.py
    ok "VAPID keys generated"
fi

# ── Step 5: API credentials ───────────────────────────────
step "5/6  Kite API credentials"
echo ""
echo -e "  ${YELLOW}Get your API key & secret from:${NC}"
echo -e "  ${BOLD}https://developers.kite.trade${NC}"
echo ""
echo -e "  (Press Enter to skip and edit config.py manually later)"
echo ""

read -rp "  Kite API Key      : " API_KEY
read -rp "  Kite API Secret   : " API_SECRET
read -rp "  Zerodha User ID   : " USER_ID

if [ -n "$API_KEY" ] && [ -n "$API_SECRET" ] && [ -n "$USER_ID" ]; then
    python3 << PYEOF
import re, sys

with open('config.py', 'r') as fh:
    txt = fh.read()

txt = re.sub(r'("api_key"\s*:\s*)"[^"]*"',    r'\g<1>"${API_KEY}"',    txt)
txt = re.sub(r'("api_secret"\s*:\s*)"[^"]*"', r'\g<1>"${API_SECRET}"', txt)
txt = re.sub(r'("user_id"\s*:\s*)"[^"]*"',    r'\g<1>"${USER_ID}"',    txt)

with open('config.py', 'w') as fh:
    fh.write(txt)

print("  config.py updated")
PYEOF
    ok "Credentials written to config.py"
else
    warn "Skipped — edit config.py manually before first run"
fi

# ── Step 6: Create launcher ───────────────────────────────
step "6/6  Create start.sh launcher"

# ── Create a quick-start script ───────────────────────────
cat > "$REPO_DIR/start.sh" << STARTEOF
#!/data/data/com.termux/files/usr/bin/bash
cd "$REPO_DIR"
SESSION="trader"
if tmux has-session -t "\$SESSION" 2>/dev/null; then
    echo "Agent already running — attaching..."
    tmux attach -t "\$SESSION"
else
    tmux new-session -d -s "\$SESSION" "python app.py; bash"
    echo "Agent started in background tmux session."
    echo "Run: tmux attach -t trader  to see logs"
    echo "Dashboard: http://localhost:8080"
fi
STARTEOF
chmod +x "$REPO_DIR/start.sh"
ok "Created start.sh shortcut"

# ── Done ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}  ════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}    Setup complete!${NC}"
echo -e "${BOLD}${GREEN}  ════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}  ❶  Confirm your Kite redirect URL is set to:${NC}"
echo -e "     ${YELLOW}http://localhost:8080/callback${NC}"
echo -e "     (Kite Developer Console → your app → Edit → Redirect URL)"
echo ""
echo -e "${BOLD}  ❷  Start the trading agent:${NC}"
echo -e "     ${YELLOW}bash ~/stock-trader-agent/start.sh${NC}"
echo -e "     (runs in background via tmux — safe to close Termux)"
echo ""
echo -e "${BOLD}  ❸  Open dashboard in Chrome:${NC}"
echo -e "     ${YELLOW}http://localhost:8080${NC}"
echo ""
echo -e "${BOLD}  ❹  Install as home screen app:${NC}"
echo -e "     Chrome menu (⋮) → Add to Home Screen"
echo ""
echo -e "${BOLD}  ❺  Log in & enable notifications in the app${NC}"
echo ""
echo -e "  tmux tips:"
echo -e "    View logs  :  ${BOLD}tmux attach -t trader${NC}"
echo -e "    Detach     :  ${BOLD}Ctrl+B  then  D${NC}"
echo -e "    Stop agent :  ${BOLD}tmux kill-session -t trader${NC}"
echo ""
