# API Reference — smart-traffic-ai
## Flask REST API v1.0

Base URL: `http://localhost:5000/api`

All protected endpoints require:
```
Authorization: Bearer <JWT_TOKEN>
```

---

## Authentication

### POST `/auth/register`
Register a new user account.

**Body:**
```json
{ "username": "alice", "email": "alice@example.com", "password": "secret123" }
```

**Response 201:**
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user": { "id": 1, "username": "alice", "email": "alice@example.com", "role": "viewer" }
}
```

---

### POST `/auth/login`
Authenticate and receive JWT token.

**Body:**
```json
{ "username": "admin", "password": "admin123" }
```

**Response 200:** Same as register.

---

### GET `/auth/me` 🔒
Return the authenticated user profile.

**Response 200:**
```json
{ "user": { "id": 1, "username": "admin", "role": "admin", "last_login": "2024-01-15T08:30:00" } }
```

---

### GET `/auth/users` 🔒 (admin only)
List all registered users.

---

## Traffic

### GET `/traffic/stats` 🔒
Dashboard KPI statistics.

**Response 200:**
```json
{
  "total_vehicles": 48392,
  "today_vehicles": 1842,
  "ambulance_events": 7,
  "active_signals": 4,
  "emergency_signals": 0,
  "density_distribution": { "low": 234, "medium": 189, "high": 97 },
  "avg_congestion_score": 0.423,
  "system_uptime_pct": 99.4
}
```

---

### GET `/traffic/density` 🔒
Live density per intersection with latest detection record.

---

### GET `/traffic/hourly` 🔒
Vehicle counts grouped by hour for the last 24 hours.

**Response 200:**
```json
{ "hourly": [{ "hour": 0, "vehicles": 34 }, { "hour": 1, "vehicles": 21 }, ...] }
```

---

### GET `/traffic/history` 🔒
Historical records with optional filters.

**Query params:**
- `intersection_id` (int)
- `hours` (int, default 24)
- `lane` (string: north|south|east|west)

---

### POST `/traffic/record` 🔒
Ingest a detection event into the database.

**Body:**
```json
{
  "intersection_id": 1,
  "lane": "north",
  "vehicle_type": "car",
  "vehicle_count": 18,
  "density_class": "medium",
  "congestion_score": 0.51,
  "ambulance_detected": false,
  "confidence": 0.94
}
```

---

## Detection

### POST `/detect/vehicles` 🔒
YOLOv8 vehicle detection. Accepts multipart file upload or base64 JSON.

**Multipart:**
```
Content-Type: multipart/form-data
Field: image (file)
```

**JSON:**
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." }
```

**Response 200:**
```json
{
  "detections": [
    { "class": "car", "confidence": 0.934, "bbox": [120, 80, 280, 200], "track_id": 3 }
  ],
  "vehicle_counts": { "car": 5, "bus": 1, "motorcycle": 2 },
  "total_vehicles": 8,
  "density_class": "low",
  "ambulance_detected": false,
  "inference_ms": 12.4,
  "mode": "yolov8"
}
```

---

### POST `/detect/ambulance` 🔒
CNN ambulance binary classification.

**Response 200:**
```json
{
  "ambulance_detected": true,
  "confidence": 0.9821,
  "action": "EMERGENCY_OVERRIDE",
  "inference_ms": 7.2,
  "model": "ambulance_cnn_v1"
}
```

---

### GET `/detect/live-feed` 🔒
Simulated live camera feed statistics.

---

## Prediction

### POST `/predict/congestion` 🔒
ML ensemble congestion class prediction.

**Body:**
```json
{
  "vehicle_count": 85,
  "car_count": 45,
  "bus_count": 8,
  "truck_count": 6,
  "motorcycle_count": 26,
  "hour": 8,
  "day_of_week": 0,
  "rain_intensity": 0.0,
  "visibility": 1.0,
  "incident_nearby": 0
}
```

**Response 200:**
```json
{
  "predicted_class": "high",
  "probabilities": { "low": 0.04, "medium": 0.11, "high": 0.85 },
  "ensemble_confidence": 0.85,
  "individual_models": {
    "random_forest":       { "class": "high", "confidence": 0.83 },
    "xgboost":             { "class": "high", "confidence": 0.87 },
    "logistic_regression": { "class": "high", "confidence": 0.76 }
  },
  "mode": "ensemble"
}
```

---

### GET `/predict/peak-hours` 🔒
24-hour congestion forecast for a given day.

**Query params:**
- `day` (int 0–6, default = today)
- `intersection_id` (int)

**Response 200:**
```json
{
  "hourly_forecast": [
    { "hour": 0, "predicted_class": "low",  "congestion_score": 0.12 },
    { "hour": 8, "predicted_class": "high", "congestion_score": 0.87 },
    ...
  ],
  "peak_hours": [{ "hour": 8 }, { "hour": 17 }, { "hour": 18 }]
}
```

---

### POST `/predict/signal-timing` 🔒
ML-optimised signal green durations.

**Body:**
```json
{
  "lane_counts": { "north": 45, "south": 20, "east": 10, "west": 30 },
  "ambulance_lane": null
}
```

**Response 200:**
```json
{
  "recommended_timings": { "north": 48, "south": 22, "east": 14, "west": 36 },
  "total_cycle_seconds": 120,
  "mode": "ml_optimized",
  "ambulance_override": false,
  "estimated_wait_reduction_pct": 34.2
}
```

---

## Analytics

### GET `/analytics/heatmap` 🔒
24×7 congestion heatmap data.

**Response 200:**
```json
{
  "heatmap": [
    { "day": "Mon", "hour": 8,  "value": 0.87 },
    { "day": "Mon", "hour": 12, "value": 0.51 },
    ...
  ]
}
```

---

### GET `/analytics/vehicle-breakdown` 🔒
Vehicle type distribution.

**Query:** `hours` (int, default 24)

---

### GET `/analytics/trends` 🔒
Daily vehicle totals for the last 7 days.

---

### GET `/analytics/congestion-history` 🔒
Congestion score time series.

**Query:** `hours` (int, default 12), `intersection_id` (int)

---

### GET `/analytics/summary` 🔒
High-level performance summary with model accuracy metrics.

---

## Signals

### GET `/signals/` 🔒
List all traffic signals with current timings and status.

---

### GET `/signals/:id` 🔒
Get a single signal by ID.

---

### PATCH `/signals/:id/update` 🔒 (operator/admin)
Update signal timings or status.

**Body:**
```json
{
  "timings": { "north": 45, "south": 25, "east": 20, "west": 30 },
  "status": "active"
}
```

---

### POST `/signals/:id/emergency` 🔒
Activate emergency override for ambulance passage.

**Body:** `{ "lane": "north" }`

**Response 200:**
```json
{
  "signal": { "status": "emergency", "emergency_lane": "north", "timings": { "north": 90, "south": 5, "east": 5, "west": 5 } },
  "message": "Emergency override activated — north lane priority"
}
```

---

### POST `/signals/:id/reset` 🔒
Reset signal to default ML-optimised state.

---

## Error Responses

All errors follow the format:
```json
{ "error": "Human-readable message", "code": 400 }
```

| Code | Meaning |
|------|---------|
| 400 | Bad request / missing fields |
| 401 | Missing or invalid JWT token |
| 403 | Insufficient role permissions |
| 404 | Resource not found |
| 409 | Conflict (e.g. duplicate username) |
| 422 | Unprocessable entity |
| 500 | Internal server error |
