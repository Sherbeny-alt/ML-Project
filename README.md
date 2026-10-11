---
title: Predictive Maintenance
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Predictive Maintenance – Machine Failure Prediction

XGBoost classifier that predicts machine failure from temperature, rotational speed,
torque, tool wear and machine type, served with a Gradio UI.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py Predictive_Maintenance_1.1.csv   # creates model.joblib
python app.py
```
