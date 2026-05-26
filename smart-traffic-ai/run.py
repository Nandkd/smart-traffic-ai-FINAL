#!/usr/bin/env python3
"""
run.py — One-command project launcher
======================================
Starts both Flask backend and Vite frontend concurrently.

Usage:
    python run.py             # start both servers
    python run.py --backend   # backend only
    python run.py --frontend  # frontend only
    python run.py --train     # run ML training pipeline
    python run.py --setup     # install all dependencies
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
ML_DIR = ROOT / "ml_models"


def run_cmd(cmd, cwd=None, label=""):
    print(f"[{label}] Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)


def setup():
    print("📦 Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)

    print("\n📦 Installing Node.js dependencies...")
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    print("\n✅ All dependencies installed!")
    print("   Run: python run.py")


def train_ml():
    print("🧠 Running ML Training Pipeline...")

    # Step 1: Generate synthetic dataset
    print("\n[1/3] Generating synthetic traffic dataset...")
    subprocess.run([
        sys.executable, "-m", "ml_models.congestion.train_models", "--generate-synthetic"
    ], cwd=ROOT, check=True)

    # Step 2: Train congestion models
    print("\n[2/3] Training congestion ensemble...")
    subprocess.run([
        sys.executable, str(ML_DIR / "congestion" / "train_models.py")
    ], cwd=ROOT, check=True)

    # Step 3: Evaluate
    print("\n[3/3] Evaluating models...")
    subprocess.run([
        sys.executable, str(ML_DIR / "congestion" / "evaluate.py")
    ], cwd=ROOT, check=True)

    print("\n✅ ML training pipeline complete!")
    print("   For YOLOv8 training: python ml_models/yolo/train.py")
    print("   For CNN training:    python ml_models/cnn/train.py")


def start_backend():
    env = os.environ.copy()
    env["FLASK_APP"] = "backend.app"
    env["FLASK_ENV"] = "development"
    return run_cmd(
        [sys.executable, "-m", "flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"],
        cwd=ROOT, label="BACKEND"
    )


def start_frontend():
    return run_cmd(["npm", "run", "dev"], cwd=FRONTEND_DIR, label="FRONTEND")


def main():
    ap = argparse.ArgumentParser(description="Smart Traffic AI — Project Launcher")
    ap.add_argument("--backend", action="store_true", help="Start backend only")
    ap.add_argument("--frontend", action="store_true", help="Start frontend only")
    ap.add_argument("--train", action="store_true", help="Run ML training")
    ap.add_argument("--setup", action="store_true", help="Install dependencies")
    args = ap.parse_args()

    if args.setup:
        setup()
        return

    if args.train:
        train_ml()
        return

    procs = []
    try:
        if args.frontend:
            procs.append(start_frontend())
        elif args.backend:
            procs.append(start_backend())
        else:
            print("🚦 Starting Smart Traffic AI System...")
            print("   Backend  → http://localhost:5000")
            print("   Frontend → http://localhost:5173")
            print("   Press Ctrl+C to stop\n")
            procs.append(start_backend())
            procs.append(start_frontend())

        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\n⏹️  Stopping all processes...")
        for p in procs:
            p.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
