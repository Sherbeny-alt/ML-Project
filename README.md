# Predictive Maintenance – Machine Failure Prediction

XGBoost classifier that predicts machine failure from temperature, rotational speed,
torque, tool wear and machine type.

Live demo (GitHub Pages): https://sherbeny-alt.github.io/ML-Project/

## Files
- `train_model.py` – trains the model and saves `model.joblib`
- `export_static.py` – exports the model to `docs/model.json` for the web app
- `docs/` – static website served by GitHub Pages (`index.html`, `icon.png`, `model.json`)
- `app.py` – optional local Gradio version

## Run
```bash
pip install -r requirements.txt
python train_model.py Predictive_Maintenance_1.1.csv   # creates model.joblib
python export_static.py                                # creates docs/model.json
python app.py                                          # optional local Gradio app
```
