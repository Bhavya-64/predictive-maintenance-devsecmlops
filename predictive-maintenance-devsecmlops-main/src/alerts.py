"""
Alert generation.

Every alert is:
  1. written to logs/alerts.jsonl (one JSON object per line - easy to audit)
  2. logged to the app log
  3. sent to a Slack/Discord/Teams webhook IF the ALERT_WEBHOOK_URL env variable is set

Alert types used in this project:
  CRITICAL_FAILURE_RISK  - API predicted CRITICAL risk for a machine
  DRIFT_DETECTED         - sensor or prediction drift found
  MODEL_RETRAINED        - new model trained and promoted
  RETRAINING_REJECTED    - new model was worse, old one kept
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

from src.utils import get_logger, path

logger = get_logger("alerts")


def send_alert(alert_type: str, severity: str, details: dict) -> dict:
    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type": alert_type,
        "severity": severity,
        "details": details,
    }
    with open(path("logs/alerts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, default=str) + "\n")
    logger.warning("ALERT [%s] %s: %s", severity.upper(), alert_type, details)

    webhook = os.getenv("ALERT_WEBHOOK_URL")
    if webhook and webhook.startswith("https://"):  # only allow HTTPS webhooks
        try:
            body = json.dumps({"text": f"[{severity.upper()}] {alert_type}: {details}"}).encode()
            req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)  # nosec B310
        except Exception as exc:  # alerting must never crash the app
            logger.error("Webhook alert failed: %s", exc)
    return alert
