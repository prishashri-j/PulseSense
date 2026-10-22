# PulseSense interactive demo

A Python + Streamlit demonstration of PulseSense's two-stage stress classification idea:

1. A frozen population-level classifier scores unseen windows.
2. A nearest-prototype layer adapts predictions from a few labeled baseline and stress examples for a simulated user.

The dashboard lets you change the simulated user's physiology, choose the number of calibration windows per class, set a stress alert threshold, and compare population-only and personalized results.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Demo notes

This prototype uses generated sensor-like data. It does not load WESAD, reproduce the reported cross-validation scores, or provide medical advice. Its lightweight distance classifier and prototype update are meant to make the project idea visible and interactive; they are not a trained neural network.
