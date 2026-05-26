"""
tests/test_api.py
==================
Unit tests for the Flask REST API endpoints.

Run: pytest tests/ -v
"""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app import create_app, db as _db


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    app = create_app("testing")
    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def auth_token(client):
    """Register + login, return JWT token."""
    client.post("/api/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "test1234",
    })
    resp = client.post("/api/auth/login", json={
        "username": "testuser",
        "password": "test1234",
    })
    data = resp.get_json()
    return data.get("token")


@pytest.fixture
def headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


# ── Auth tests ─────────────────────────────────────────────────

class TestAuth:
    def test_register_success(self, client):
        resp = client.post("/api/auth/register", json={
            "username": "newuser99",
            "email": "newuser99@test.com",
            "password": "password99",
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert "token" in data
        assert data["user"]["username"] == "newuser99"

    def test_register_duplicate(self, client):
        body = {"username": "dupuser", "email": "dup@test.com", "password": "pass1234"}
        client.post("/api/auth/register", json=body)
        resp = client.post("/api/auth/register", json=body)
        assert resp.status_code == 409

    def test_login_success(self, client):
        client.post("/api/auth/register", json={
            "username": "logintest", "email": "lt@t.com", "password": "ltpass123"
        })
        resp = client.post("/api/auth/login", json={
            "username": "logintest", "password": "ltpass123"
        })
        assert resp.status_code == 200
        assert "token" in resp.get_json()

    def test_login_wrong_password(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "testuser", "password": "wrongpass"
        })
        assert resp.status_code == 401

    def test_me_authenticated(self, client, headers):
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 200
        assert "user" in resp.get_json()

    def test_me_unauthenticated(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


# ── Health check ───────────────────────────────────────────────

class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# ── Traffic API ────────────────────────────────────────────────

class TestTrafficAPI:
    def test_get_stats(self, client, headers):
        resp = client.get("/api/traffic/stats", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_vehicles" in data
        assert "active_signals" in data

    def test_get_hourly(self, client, headers):
        resp = client.get("/api/traffic/hourly", headers=headers)
        assert resp.status_code == 200
        assert "hourly" in resp.get_json()

    def test_get_density(self, client, headers):
        resp = client.get("/api/traffic/density", headers=headers)
        assert resp.status_code == 200
        assert "data" in resp.get_json()

    def test_add_record(self, client, headers):
        resp = client.post("/api/traffic/record", headers=headers, json={
            "intersection_id": 1,
            "lane": "north",
            "vehicle_type": "car",
            "vehicle_count": 15,
            "density_class": "medium",
            "congestion_score": 0.42,
            "ambulance_detected": False,
            "confidence": 0.91,
        })
        assert resp.status_code == 201
        assert resp.get_json()["record"]["vehicle_count"] == 15


# ── Prediction API ─────────────────────────────────────────────

class TestPredictionAPI:
    def test_predict_congestion(self, client, headers):
        resp = client.post("/api/predict/congestion", headers=headers, json={
            "vehicle_count": 90,
            "hour": 8,
            "day_of_week": 0,
            "rain_intensity": 0.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ("low", "medium", "high")
        assert "probabilities" in data

    def test_peak_hours(self, client, headers):
        resp = client.get("/api/predict/peak-hours?day=0", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["hourly_forecast"]) == 24

    def test_signal_timing_optimizer(self, client, headers):
        resp = client.post("/api/predict/signal-timing", headers=headers, json={
            "lane_counts": {"north": 40, "south": 20, "east": 10, "west": 30}
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommended_timings" in data
        timings = data["recommended_timings"]
        assert all(t >= 5 for t in timings.values())

    def test_emergency_override_timing(self, client, headers):
        resp = client.post("/api/predict/signal-timing", headers=headers, json={
            "lane_counts": {"north": 40, "south": 20, "east": 10, "west": 30},
            "ambulance_lane": "north",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ambulance_override"] is True
        assert data["recommended_timings"]["north"] == 90


# ── Analytics API ──────────────────────────────────────────────

class TestAnalyticsAPI:
    def test_heatmap(self, client, headers):
        resp = client.get("/api/analytics/heatmap", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "heatmap" in data
        assert len(data["heatmap"]) == 168  # 7 days × 24 hours

    def test_trends(self, client, headers):
        resp = client.get("/api/analytics/trends", headers=headers)
        assert resp.status_code == 200
        assert "trends" in resp.get_json()

    def test_vehicle_breakdown(self, client, headers):
        resp = client.get("/api/analytics/vehicle-breakdown", headers=headers)
        assert resp.status_code == 200
        assert "breakdown" in resp.get_json()

    def test_summary(self, client, headers):
        resp = client.get("/api/analytics/summary", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "avg_wait_reduction_pct" in data


# ── Signals API ────────────────────────────────────────────────

class TestSignalsAPI:
    def test_list_signals(self, client, headers):
        resp = client.get("/api/signals/", headers=headers)
        assert resp.status_code == 200
        assert "signals" in resp.get_json()

    def test_get_signal(self, client, headers):
        resp = client.get("/api/signals/1", headers=headers)
        assert resp.status_code == 200
        sig = resp.get_json()["signal"]
        assert "location_name" in sig
        assert "timings" in sig

    def test_emergency_override(self, client, headers):
        resp = client.post("/api/signals/1/emergency", headers=headers,
                           json={"lane": "south"})
        assert resp.status_code == 200
        sig = resp.get_json()["signal"]
        assert sig["status"] == "emergency"
        assert sig["timings"]["south"] == 90

    def test_reset_signal(self, client, headers):
        resp = client.post("/api/signals/1/reset", headers=headers)
        assert resp.status_code == 200
        assert resp.get_json()["signal"]["status"] == "active"
