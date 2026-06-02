"""
telemetry_dashboard_demo.py
============================
Async Python demo — shows how a UI dashboard subscribes to the
Crossroad AI TELEMETRY_UPDATE stream and prints a live vehicle-count table.

Run standalone (no server needed — uses asyncio.Queue to mock SSE):
    python telemetry_dashboard_demo.py

To connect to the real backend instead, swap queue.get() for the
aiohttp SSE snippet shown in `dashboard_subscriber()`.

JSON Event Schema
-----------------
TELEMETRY_UPDATE:
{
  "event": "TELEMETRY_UPDATE",
  "intersection_state": {
    "north": {
      "lane":   "North",
      "counts": {"car": 2, "motorcycle": 10, "auto_rickshaw": 5, "bus": 1},
      "total":  18,
      "alert":  null
    },
    "south": { "lane": "South", "counts": {...}, "total": 4,  "alert": null },
    "east":  { "lane": "East",  "counts": {...}, "total": 7,  "alert": null },
    "west":  { "lane": "West",  "counts": {...}, "total": 3,  "alert": null }
  },
  "emergency":      false,
  "emergency_road": null,
  "signal_mode":    "auto",
  "timestamp":      "2025-06-02T12:34:56.789012"
}

EMERGENCY_DETECTED — same shape but:
  "event": "EMERGENCY_DETECTED"
  "emergency": true
  "emergency_road": "north"
  "alert": "Ambulance in North Lane"
  and intersection_state["north"]["alert"] = "Ambulance in North Lane"
"""

import asyncio
import random
from datetime import datetime


# ── Simulated YOLO sense-module output ───────────────────────────
LANES   = ["north", "south", "east", "west"]
V_TYPES = ["car", "motorcycle", "auto_rickshaw", "bus", "truck", "bicycle"]


def _mock_lane_counts(lane: str, tick: int) -> dict:
    """Seed-based random — same tick always produces the same counts (reproducible)."""
    rng = random.Random(hash(lane) ^ (tick * 2654435761))
    return {
        "car":           rng.randint(1, 8),
        "motorcycle":    rng.randint(3, 15),
        "auto_rickshaw": rng.randint(1, 6),
        "bus":           rng.randint(0, 3),
        "truck":         rng.randint(0, 2),
        "bicycle":       rng.randint(0, 4),
    }


# ── Mock TrafficOrchestrator — mirrors the real telemetry broadcaster ──
async def mock_telemetry_producer(queue: asyncio.Queue) -> None:
    """
    Mimics the backend broadcasting intersection_state every 1 second
    (the same data the SSE endpoint at /api/crossroad/telemetry/stream delivers).

    Tick 8 injects an EMERGENCY_DETECTED event so you can see the alert flash.
    """
    tick = 0
    EMERGENCY_TICK = 8   # trigger ambulance at t=8 s

    while True:
        is_emergency = (tick == EMERGENCY_TICK)
        emerg_road   = "north" if is_emergency else None
        sig_mode     = "emergency" if is_emergency else "auto"

        intersection_state: dict = {}
        for lane in LANES:
            counts  = _mock_lane_counts(lane, tick)
            total   = sum(counts.values())
            is_amb  = (lane == emerg_road)
            alert   = f"Ambulance in {lane.capitalize()} Lane" if is_amb else None
            intersection_state[lane] = {
                "lane":   lane.capitalize(),
                "counts": counts,
                "total":  total,
                "alert":  alert,
            }

        event_type = "EMERGENCY_DETECTED" if is_emergency else "TELEMETRY_UPDATE"
        payload: dict = {
            "event":              event_type,
            "intersection_state": intersection_state,
            "emergency":          is_emergency,
            "emergency_road":     emerg_road,
            "signal_mode":        sig_mode,
            "timestamp":          datetime.utcnow().isoformat(),
        }
        if is_emergency and emerg_road:
            payload["alert"] = f"Ambulance in {emerg_road.capitalize()} Lane"

        await queue.put(payload)
        tick += 1
        await asyncio.sleep(1.0)


# ── Dashboard renderer ───────────────────────────────────────────
def _render_table(payload: dict, tick: int) -> None:
    """Print a live vehicle-count table — clears screen each update."""
    state      = payload["intersection_state"]
    emergency  = payload.get("emergency", False)
    emerg_road = payload.get("emergency_road")
    ts         = payload.get("timestamp", "")[:19]
    event      = payload.get("event", "TELEMETRY_UPDATE")
    sig_mode   = payload.get("signal_mode", "auto").upper()

    # ANSI clear-screen (works on Windows 10+ and all Unix terminals)
    print("\033[H\033[J", end="")

    # ── Header ──────────────────────────────────────────────────
    print("=" * 66)
    print(f"  CROSSROAD AI  ·  LIVE TELEMETRY DASHBOARD        [{ts}]")
    print("=" * 66)

    # ── Emergency alert banner ───────────────────────────────────
    if emergency and emerg_road:
        alert_msg = payload.get("alert", f"Ambulance in {emerg_road.capitalize()} Lane")
        # Blinking red on terminals that support ANSI
        blink = "\033[5;31m"
        reset = "\033[0m"
        print(f"\n  {blink}⚠  EMERGENCY ALERT: {alert_msg}  ⚠{reset}\n")
    else:
        print()

    # ── Table header ─────────────────────────────────────────────
    print(f"  {'Lane':<10} {'Cars':>5} {'Moto':>6} {'Auto':>6} {'Bus':>5} {'Truck':>6} {'Bike':>6} {'TOTAL':>7}")
    print("  " + "─" * 52)

    col_sums   = dict.fromkeys(["car", "motorcycle", "auto_rickshaw", "bus", "truck", "bicycle"], 0)
    grand_total = 0

    for lane_key in LANES:
        d     = state.get(lane_key, {})
        name  = d.get("lane", lane_key.capitalize())
        cnts  = d.get("counts", {})
        total = d.get("total", 0)
        alert = d.get("alert")

        # Prefix: 🚨 for alert lanes, space otherwise
        prefix = "🚨" if alert else "  "
        suffix = f"  ← {alert}" if alert else ""

        print(
            f"{prefix} {name:<9}"
            f" {cnts.get('car', 0):>5}"
            f" {cnts.get('motorcycle', 0):>6}"
            f" {cnts.get('auto_rickshaw', 0):>6}"
            f" {cnts.get('bus', 0):>5}"
            f" {cnts.get('truck', 0):>6}"
            f" {cnts.get('bicycle', 0):>6}"
            f" {total:>7}"
            f"{suffix}"
        )

        grand_total += total
        for k in col_sums:
            col_sums[k] += cnts.get(k, 0)

    print("  " + "─" * 52)
    print(
        f"  {'TOTAL':<10}"
        f" {col_sums['car']:>5}"
        f" {col_sums['motorcycle']:>6}"
        f" {col_sums['auto_rickshaw']:>6}"
        f" {col_sums['bus']:>5}"
        f" {col_sums['truck']:>6}"
        f" {col_sums['bicycle']:>6}"
        f" {grand_total:>7}"
    )

    # ── Footer ───────────────────────────────────────────────────
    print("=" * 66)
    print(f"  Event: {event:<30} Mode: {sig_mode}")
    print(f"  Tick:  {tick:<5}  (t=8 triggers EMERGENCY_DETECTED demo)")
    print("  Press Ctrl+C to stop.")


# ── Dashboard subscriber ─────────────────────────────────────────
async def dashboard_subscriber(queue: asyncio.Queue) -> None:
    """
    Consumes telemetry events from the queue and renders the live table.

    ── Production replacement (aiohttp SSE) ──────────────────────
    Replace the queue.get() loop below with:

        import aiohttp, json

        API   = "http://localhost:5000/api"
        TOKEN = "<your JWT>"

        async with aiohttp.ClientSession() as session:
            url = f"{API}/crossroad/telemetry/stream?jwt={TOKEN}"
            async with session.get(url) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        _render_table(payload, tick)
                        tick += 1
    ──────────────────────────────────────────────────────────────
    """
    print("Dashboard subscriber connected — waiting for telemetry...\n")
    await asyncio.sleep(0.3)  # let the producer send the first event

    tick = 0
    while True:
        payload = await queue.get()
        _render_table(payload, tick)
        queue.task_done()
        tick += 1


# ── Entry point ──────────────────────────────────────────────────
async def main() -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    await asyncio.gather(
        mock_telemetry_producer(queue),
        dashboard_subscriber(queue),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDashboard disconnected.")
