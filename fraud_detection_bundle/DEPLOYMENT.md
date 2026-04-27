# Deployment guide

The app is a FastAPI **web service** that serves the SPA at `/` and an
HTTP API at `/api/*`. Pick one of the recipes below; each one is
independently complete.

## Configuration

All optional. The service runs in dev mode with auth off if you set none.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Enables Claude vision corroboration. Without it, runs the deterministic offline fusion. |
| `DEEPSEEK_API_KEY` | — | Enables DeepSeek text-only reasoning over the detector evidence. |
| `FRAUD_AUTH_TOKEN` | — | When set, `/api/analyze` requires `Authorization: Bearer <token>`. |
| `FRAUD_ALLOWED_ORIGINS` | — | Comma-separated CORS origins, e.g. `https://console.mythosbank.com`. |
| `FRAUD_MAX_BODY_MB` | `50` | Per-request body cap. |
| `FRAUD_REPORT_DIR` | — | If set, every report is persisted as `<dir>/<reproducibility_hash>.json` for audit. |
| `FRAUD_LOG_LEVEL` | `INFO` | Standard Python logging level. |
| `FRAUD_ENABLE_DOCS` | — | When set, exposes `/api/docs` (Swagger UI). Off by default in production. |

Generate a token: `openssl rand -hex 32`.

## Picking a provider

The web UI's Settings drawer lets each user pick **Anthropic Claude**
(vision-capable), **DeepSeek** (text-only), or **Offline** (deterministic
local fusion, no LLM). For deployments that should *force* a single
provider, set its API key as an env var on the host — that disables
per-user overrides for audit-trail integrity.

| Use case | Set this |
| --- | --- |
| Vision corroboration matters (KYC selfies, liveness clips, doctored proof-of-address) | `ANTHROPIC_API_KEY` |
| You only have a DeepSeek key and want LLM reasoning over detector evidence | `DEEPSEEK_API_KEY` |
| Self-serve deployment where each user supplies their own key | leave both unset |
| Air-gapped / offline deployment | leave both unset; users pick "Offline" in Settings |

## Recipes

### 1. Local Docker (one-shot, with persistence)

```bash
# Pick ONE provider key (or omit and let users pick in the UI):
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
# OR
echo "DEEPSEEK_API_KEY=sk-..." > .env

echo "FRAUD_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env
docker compose up --build -d
docker compose logs -f
```

Open `http://localhost:8000`.

### 2. Fly.io

```bash
fly launch --no-deploy --copy-config --name mythos-fraud-detection
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly secrets set FRAUD_AUTH_TOKEN=$(openssl rand -hex 32)
fly volumes create fraud_reports --size 1
fly deploy
fly open
```

`fly.toml` is pre-configured: 1 GB RAM, shared CPU, auto-stop when idle,
TLS termination, `/api/health` checks every 30 s.

### 3. Render.com (Blueprint)

Push the repo to GitHub, then in Render: **New + → Blueprint →** point at
the repo. Render reads `render.yaml`, which provisions the service plus
a 1 GB persistent disk for reports. Add `ANTHROPIC_API_KEY` and
`FRAUD_AUTH_TOKEN` as encrypted env vars in the dashboard.

### 4. Railway / Heroku (Procfile)

`Procfile` is set up. On Railway:

```bash
railway login
railway init
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set FRAUD_AUTH_TOKEN=$(openssl rand -hex 32)
railway up
```

### 5. AWS (ECR + ECS Fargate or App Runner)

Build & push:

```bash
aws ecr create-repository --repository-name mythos-fraud-detection
$(aws ecr get-login --no-include-email)
docker build -t mythos-fraud-detection .
docker tag mythos-fraud-detection:latest <acct>.dkr.ecr.<region>.amazonaws.com/mythos-fraud-detection:0.2.0
docker push <acct>.dkr.ecr.<region>.amazonaws.com/mythos-fraud-detection:0.2.0
```

Then point App Runner at the ECR image, set env vars, and attach an
ALB if you need a custom domain. For ECS Fargate, mount an EFS share at
`/var/lib/fraud_reports` for the audit trail.

### 6. Bare-metal / VPS (systemd + nginx)

```bash
# On the host, as root:
adduser --system --group --home /opt/mythos-fraud-detection mythos
sudo -u mythos git clone https://github.com/ramishaheen/MythosBanking /opt/mythos-fraud-detection
cd /opt/mythos-fraud-detection
sudo -u mythos python3 -m venv .venv
sudo -u mythos .venv/bin/pip install -e .[all]
sudo -u mythos .venv/bin/pip install -r web/requirements.txt
mkdir -p /var/lib/fraud_reports && chown mythos:mythos /var/lib/fraud_reports

# Secrets file (mode 600):
cat > /etc/mythos-fraud-detection.env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
FRAUD_AUTH_TOKEN=...replace...
FRAUD_ALLOWED_ORIGINS=https://console.mythosbank.example.com
FRAUD_MAX_BODY_MB=50
FRAUD_REPORT_DIR=/var/lib/fraud_reports
EOF
chmod 600 /etc/mythos-fraud-detection.env

cp deploy/systemd/mythos-fraud-detection.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now mythos-fraud-detection

cp deploy/nginx/mythos-fraud-detection.conf /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/mythos-fraud-detection.conf /etc/nginx/sites-enabled/
certbot --nginx -d console.mythosbank.example.com
systemctl reload nginx
```

The systemd unit hardens the process (NoNewPrivileges, ProtectSystem,
PrivateTmp, ReadWritePaths-only, MemoryDenyWriteExecute). The nginx
config terminates TLS, sets HSTS, caps body size at 50 MB, and forwards
to the app on `127.0.0.1:8000`.

## Smoke test (any environment)

```bash
curl https://YOUR-HOST/api/health
# {"status":"ok","version":"0.2.0","model":"claude-sonnet-4-6","online":true,
#  "auth_required":true,"max_body_mb":50,"reports_persisted":true}

curl -H "Authorization: Bearer $FRAUD_AUTH_TOKEN" \
     -F files=@sample.jpg -F kinds=image \
     https://YOUR-HOST/api/analyze | jq .verdict
```

## Operational checklist

Before opening the service to real traffic:

- [ ] `FRAUD_AUTH_TOKEN` set (or front the service with your own auth).
- [ ] `FRAUD_ALLOWED_ORIGINS` set if the SPA is on a different host.
- [ ] TLS certificate provisioned (Let's Encrypt via certbot, or your CDN).
- [ ] `FRAUD_REPORT_DIR` mounted on durable storage; reports backed up.
- [ ] Log shipping configured (the app emits one JSON line per request).
- [ ] Body-size cap enforced at the gateway in addition to the app.
- [ ] Rate limiting at the gateway (e.g. nginx `limit_req` zone).
- [ ] Periodic re-calibration: re-run `python -m fraud_detection.calibration`
      against a labelled production dataset; commit the resulting JSON
      with a date stamp so the chain-of-custody is auditable.
