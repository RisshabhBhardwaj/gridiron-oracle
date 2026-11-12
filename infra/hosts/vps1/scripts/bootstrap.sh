#!/bin/bash
# Fresh-Ubuntu (24.04) bootstrap for Gridiron VPS-1 — idempotent.
#
# Run as root on first boot, re-runnable after that (every step checks state
# first). Covers PLATFORM_SETUP §4 as applied to a TUNNEL-ONLY box:
#   - SSH: `deploy` user + key, root login off, password auth off.
#   - Firewall: deny incoming, allow 22/tcp ONLY (no 80/443 — there is no
#     public listener on this box; ingress arrives via Cloudflare Tunnel).
#   - unattended-upgrades (security, reboot OFF — monthly manual window).
#   - 4 GB swapfile, low swappiness (safety valve, not performance).
#   - Docker Engine + compose plugin from Docker's apt repo; daemon-wide
#     log rotation + live-restore.
#
# Usage:
#   SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" bash scripts/bootstrap.sh
#   # or: bash scripts/bootstrap.sh "$(cat ~/.ssh/id_ed25519.pub)"
#
# IMPORTANT — read before running:
#   1. SSH hardening refuses to disable password auth until `deploy` holds a
#      non-empty authorized_keys (lockout guard). Run with a key, or install
#      one first and re-run.
#   2. ufw allows ONLY port 22. If your client IP changes you need OVH's
#      KVM/rescue console. That is the intended trade-off for a box with no
#      public listeners (§4 allows IP-pinning; tunnel-only makes it safe).
#   3. After hardening, open a NEW SSH session as deploy and confirm it
#      works BEFORE closing this one. The script reminds you at the end.
set -euo pipefail

PUBKEY="${1:-${SSH_PUBKEY:-}}"
DEPLOY_USER="deploy"

log() { echo "bootstrap: $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── 0. Preconditions ─────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
  echo "bootstrap: run as root (first boot) — aborting" >&2
  exit 1
fi

# ── 1. deploy user + SSH key ─────────────────────────────────────────────────
if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
  log "creating user $DEPLOY_USER ..."
  useradd --create-home --shell /bin/bash "$DEPLOY_USER"
  usermod -aG sudo "$DEPLOY_USER"
else
  log "user $DEPLOY_USER exists — ensuring sudo ..."
  usermod -aG sudo "$DEPLOY_USER"
fi

SSH_DIR="/home/$DEPLOY_USER/.ssh"
AUTH_KEYS="$SSH_DIR/authorized_keys"
mkdir -p "$SSH_DIR"
touch "$AUTH_KEYS"
if [ -n "$PUBKEY" ]; then
  if ! grep -qxF "$PUBKEY" "$AUTH_KEYS" 2>/dev/null; then
    echo "$PUBKEY" >> "$AUTH_KEYS"
    log "installed SSH public key for $DEPLOY_USER"
  else
    log "SSH public key already present — skipping"
  fi
else
  log "WARNING: no key supplied (arg 1 or SSH_PUBKEY) — SSH hardening will be skipped until a key exists"
fi
chmod 700 "$SSH_DIR"
chmod 600 "$AUTH_KEYS"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$SSH_DIR"

# ── 2. SSH hardening (lockout-guarded) ───────────────────────────────────────
HARDEN_CONF="/etc/ssh/sshd_config.d/10-hardening.conf"
if [ -s "$AUTH_KEYS" ]; then
  log "writing $HARDEN_CONF ..."
  cat > "$HARDEN_CONF" <<'EOF'
# Gridiron VPS-1 hardening (bootstrap.sh managed).
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
AllowUsers deploy
MaxAuthTries 3
EOF
  chmod 644 "$HARDEN_CONF"
  sshd -t
  if have systemctl; then
    systemctl reload sshd || systemctl reload ssh || true
  else
    service ssh reload || true
  fi
  log "sshd reloaded with hardening"
else
  log "SKIPPING sshd hardening: $AUTH_KEYS is empty (would lock you out)."
  log "Re-run with a key: SSH_PUBKEY=\"\$(cat ~/.ssh/id_ed25519.pub)\" bash $0"
fi

# ── 3. Firewall: deny incoming, allow 22 only ────────────────────────────────
if have ufw; then
  log "configuring ufw (tunnel-only: 22/tcp only) ..."
  ufw --force reset >/dev/null
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp comment 'SSH only - tunnel ingress, no public web ports'
  # NOTE: deliberately NO 80/443 rules. Nothing listens on them; Cloudflare
  # Tunnel connects outbound. Verify from outside later: only 22/tcp open.
  ufw --force enable
  ufw status verbose || true
else
  log "installing ufw ..."
  apt-get update && apt-get install -y ufw
  ufw --force reset >/dev/null
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp comment 'SSH only - tunnel ingress, no public web ports'
  ufw --force enable
fi

# fail2ban for sshd (cheap, standard).
if ! have fail2ban-server; then
  log "installing fail2ban ..."
  apt-get update && apt-get install -y fail2ban
fi
if have systemctl; then
  systemctl enable --now fail2ban || true
fi

# ── 4. Time + unattended upgrades (reboot OFF) ───────────────────────────────
if have timedatectl; then
  timedatectl set-timezone UTC || true
  if have systemctl; then
    systemctl enable --now systemd-timesyncd || true
  fi
fi
if ! have unattended-upgrade; then
  log "installing unattended-upgrades ..."
  apt-get update && apt-get install -y unattended-upgrades
fi
# Security updates automatic; everything else + reboots stay manual (monthly
# maintenance window with `needrestart`, per §4).
cat > /etc/apt/apt.conf.d/51-gridiron-unattended <<'EOF'
// Gridiron VPS-1 (bootstrap.sh managed): security only, never reboot.
Unattended-Upgrade::Allowed-Origins {
  "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::Automatic-Reboot "false";
EOF
chmod 644 /etc/apt/apt.conf.d/51-gridiron-unattended

# ── 5. Swap: 4 GB safety valve, low swappiness ────────────────────────────────
if [ ! -f /swapfile ]; then
  log "creating 4 GB swapfile ..."
  fallocate -l 4G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=4096 status=none
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  if ! grep -q '^/swapfile' /etc/fstab; then
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
  fi
else
  log "swapfile exists — ensuring active ..."
  swapon --show | grep -q '/swapfile' || swapon /swapfile || true
fi
# Low swappiness: swap is an OOM safety valve, not performance (§4).
sysctl -w vm.swappiness=10 >/dev/null
if ! grep -q '^vm.swappiness' /etc/sysctl.conf; then
  echo 'vm.swappiness=10' >> /etc/sysctl.conf
else
  sed -i 's/^vm.swappiness.*/vm.swappiness=10/' /etc/sysctl.conf
fi

# ── 6. Docker Engine + compose plugin (Docker's apt repo, not distro) ────────
if ! have docker; then
  log "installing Docker Engine from Docker's apt repo ..."
  apt-get update && apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod 644 /etc/apt/keyrings/docker.gpg
  fi
  if [ ! -f /etc/apt/sources.list.d/docker.list ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list
  fi
  apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
else
  log "docker present ($(docker --version)) — ensuring compose plugin ..."
  apt-get update && apt-get install -y docker-compose-plugin || true
fi
if have systemctl; then
  systemctl enable --now docker || true
fi
usermod -aG docker "$DEPLOY_USER" || true

# Daemon-wide log rotation + live-restore (per-service rotation also set in
# compose.yml; the daemon default is the backstop).
mkdir -p /etc/docker
if [ ! -f /etc/docker/daemon.json ]; then
  log "writing /etc/docker/daemon.json ..."
  cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "5"},
  "live-restore": true
}
EOF
  if have systemctl; then
    systemctl restart docker || true
  fi
else
  log "/etc/docker/daemon.json exists — leaving untouched (check it matches: rotation 10m x5, live-restore)"
fi

# ── 7. Sanity + handoff ──────────────────────────────────────────────────────
log "docker: $(docker run --rm hello-world 2>&1 | head -1 || echo 'hello-world FAILED')"
log "ufw: $(ufw status | head -1)"
log "swap: $(swapon --show=SIZE,USED --noheadings | tr '\n' ' ')"
log "timezone: $(timedatectl show --property=Timezone --value 2>/dev/null || date -u +%Z)"

echo
echo 'bootstrap: DONE. Before closing THIS session:'
echo "  ssh ${DEPLOY_USER}@<vps-ip>   # open a NEW session and confirm it works"
echo '  Only then close this one. Root login and password auth are now off.'
