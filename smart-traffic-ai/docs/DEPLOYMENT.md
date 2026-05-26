# Deployment Guide — smart-traffic-ai

## 1. Local Development

### Prerequisites
| Tool | Minimum Version |
|------|----------------|
| Python | 3.10 |
| Node.js | 18 |
| pip | 23 |
| npm | 9 |
| CUDA (optional) | 11.8 |

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/smart-traffic-ai.git
cd smart-traffic-ai

# One-command setup
make setup          # installs Python + Node dependencies

# Start everything
make dev            # backend :5000 + frontend :5173
```

Or manually:
```bash
# Terminal 1 — Backend
cd smart-traffic-ai
pip install -r requirements.txt
python run.py --backend

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Access:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000/api/health
- **Demo login**: `admin` / `admin123`

---

## 2. ML Training Pipeline

Run in order:

```bash
# Step 1 — Prepare synthetic datasets
make dataset
# or: python datasets/prepare_dataset.py --task all

# Step 2 — Train congestion ensemble (RF + XGBoost + LR)
make train-congestion
# or: python ml_models/congestion/train_models.py --generate-synthetic

# Step 3 — Evaluate congestion models
make evaluate
# or: python ml_models/congestion/evaluate.py

# Step 4 — Train YOLOv8 (needs real dataset)
make train-yolo
# or: python ml_models/yolo/train.py --model yolov8s --epochs 100

# Step 5 — Train CNN ambulance classifier
make train-cnn
# or: python ml_models/cnn/train.py --epochs 50
```

Expected outputs in `ml_models/weights/`:
```
congestion_ensemble.pkl   ← voting ensemble
random_forest.pkl
xgboost.pkl
logistic_regression.pkl
yolov8_traffic.pt         ← after YOLO training
ambulance_cnn.pth         ← after CNN training
```

---

## 3. Docker Deployment

### Single command (recommended for demo)
```bash
# Build and start all services
docker-compose up --build

# Access at http://localhost
```

### Individual containers
```bash
# Backend only
docker build -t traffic-backend .
docker run -p 5000:5000 -v $(pwd)/ml_models/weights:/app/ml_models/weights traffic-backend

# Frontend only
docker build -t traffic-frontend ./frontend
docker run -p 80:80 traffic-frontend
```

---

## 4. Production Deployment

### Environment variables
Copy `.env.example` to `.env` and set:
```bash
SECRET_KEY=<long-random-string>
JWT_SECRET_KEY=<another-long-random-string>
FLASK_ENV=production
```

### Gunicorn (backend)
```bash
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile - \
  "backend.app:create_app()"
```

### Nginx (frontend)
```bash
cd frontend && npm run build
# Point Nginx root to: frontend/dist/
# Use nginx.conf from frontend/ as reference
```

### Systemd service (Linux)
Create `/etc/systemd/system/traffic-ai.service`:
```ini
[Unit]
Description=Traffic AI Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/smart-traffic-ai
EnvironmentFile=/opt/smart-traffic-ai/.env
ExecStart=/opt/smart-traffic-ai/venv/bin/gunicorn \
    --workers 4 --bind 0.0.0.0:5000 \
    "backend.app:create_app()"
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable traffic-ai
sudo systemctl start traffic-ai
```

---

## 5. Real Camera Integration

Replace the simulated detection with a real RTSP stream:

```python
# In ml_models/yolo/detect.py
source = "rtsp://admin:password@192.168.1.100:554/stream1"
run(source, show=False, save=True)
```

Set in `.env`:
```
RTSP_CAMERA_1=rtsp://admin:password@192.168.1.100:554/stream
```

---

## 6. Database Migration

```bash
# Initialise migrations
flask db init

# Create migration after model changes
flask db migrate -m "Add new field"

# Apply migration
flask db upgrade

# Roll back
flask db downgrade
```

---

## 7. Running Tests

```bash
# All tests
pytest tests/ -v

# API tests only
pytest tests/test_api.py -v

# ML model tests only
pytest tests/test_ml_models.py -v

# With coverage
pytest tests/ --cov=backend --cov=ml_models --cov-report=html
```

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `CUDA out of memory` | Reduce `--batch` in train.py to 8 |
| `sqlite3.OperationalError` | Delete `traffic_system.db` and restart |
| Frontend 404 on refresh | Configure Nginx `try_files $uri /index.html` |
| JWT 401 errors | Check `JWT_SECRET_KEY` in `.env` |
| CORS errors | Add frontend URL to `CORS_ORIGINS` in `.env` |
| Model weights missing | Run `make train` first |
