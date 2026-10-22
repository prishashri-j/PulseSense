"""PulseSense interactive demo — simulated wearable data, not a medical device."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="PulseSense | Adaptive Stress Demo", page_icon="🫀", layout="wide")

FEATURES = ["Heart rate", "EDA", "Skin temperature", "Motion"]
UNITS = ["bpm", "µS", "°C", "g"]
BASELINE_MEAN = np.array([72.0, 1.15, 33.1, 0.13])
STRESS_MEAN = np.array([101.0, 3.25, 32.4, 0.24])
SCALE = np.array([13.0, 1.25, 0.65, 0.12])


def make_subject(seed: int, shift: np.ndarray, count: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a person's windows with individual baseline offsets and overlap."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1]), count // 2)
    rng.shuffle(labels)
    # Individual resting physiology shifts all channels. Stress response also varies.
    offset = shift
    response = np.array([1.0, 1.0, 1.0, 1.0]) + rng.normal(0, 0.14, 4)
    data = np.empty((count, 4))
    for i, label in enumerate(labels):
        center = BASELINE_MEAN.copy() if label == 0 else BASELINE_MEAN + (STRESS_MEAN - BASELINE_MEAN) * response
        data[i] = center + offset + rng.normal(0, SCALE * 0.78)
    return data, labels


def base_scores(x: np.ndarray) -> np.ndarray:
    """A fixed population-level distance classifier, used as the frozen backbone."""
    # Scale features before distance calculations; map distances to soft confidence.
    base = ((x - BASELINE_MEAN) / SCALE) ** 2
    stress = ((x - STRESS_MEAN) / SCALE) ** 2
    logits = np.stack([-base.sum(axis=1), -stress.sum(axis=1)], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def prototype_scores(x: np.ndarray, support_x: np.ndarray, support_y: np.ndarray) -> np.ndarray:
    """Nearest-class-prototype probabilities in normalized feature space."""
    prototypes = np.stack([support_x[support_y == c].mean(axis=0) for c in (0, 1)])
    dist = np.stack([(((x - p) / SCALE) ** 2).mean(axis=1) for p in prototypes], axis=1)
    logits = -dist / 0.8
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def metric_card(label: str, value: str, detail: str, color: str = "#6ee7b7") -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Manrope', sans-serif;}
.stApp {background: radial-gradient(ellipse at 15% 0%, #112c38 0, #09141e 45%, #080e16 100%); color:#e6f1f5;}
.block-container {max-width: 1440px; padding-top: 2rem;}
.hero {display:flex;justify-content:space-between;align-items:center;margin-bottom:1.4rem;}
.brand {font-size:13px;letter-spacing:2.5px;color:#7dd3c7;font-weight:800;text-transform:uppercase;}
.hero h1 {margin:.35rem 0 0;font-size:34px;letter-spacing:-1.5px;color:#f0f8fa;}
.hero p {color:#8da6b2;margin:.4rem 0 0;font-size:14px;}
.live {border:1px solid #245848;background:#12372f;color:#91f2c5;border-radius:99px;padding:9px 14px;font-size:11px;letter-spacing:1.5px;font-family:'DM Mono',monospace;}
.metric {background:linear-gradient(145deg,rgba(19,39,51,.94),rgba(13,25,35,.94));border:1px solid #1f3946;border-radius:14px;padding:18px 20px;min-height:112px;}
.metric-label {font-size:11px;color:#89a4b0;text-transform:uppercase;letter-spacing:1.4px;}
.metric-value {font-size:27px;font-weight:800;margin:8px 0 1px;}
.metric-detail {font-size:11px;color:#728c98;}
.panel {background:rgba(13,26,36,.8);border:1px solid #1e3744;border-radius:15px;padding:18px 20px;margin-bottom:14px;}
.eyebrow {font:11px 'DM Mono',monospace;color:#6dd4c1;text-transform:uppercase;letter-spacing:1.5px;}
.section-title {font-size:17px;font-weight:700;color:#e9f3f6;margin:6px 0 2px;}
.muted {font-size:12px;color:#849ba6;}
.step {background:#10252e;border:1px solid #22434c;border-radius:10px;padding:13px;margin:8px 0;color:#c4d6dc;font-size:12px;}
.step b {color:#79dec3;}
div[data-testid="stMetric"] {background:#10232e;border:1px solid #1c3946;padding:12px;border-radius:11px;}
[data-testid="stSidebar"] {background:#0b1720;border-right:1px solid #1b303c;}
.stButton>button {background:#46c6a2;color:#062019;border:0;border-radius:9px;font-weight:800;padding:.65rem 1rem;}
.stButton>button:hover {background:#71e0ba;color:#062019;border:0;}
</style>""", unsafe_allow_html=True)

if "seed" not in st.session_state:
    st.session_state.seed = 41

with st.sidebar:
    st.markdown('<div class="eyebrow">Demo controls</div>', unsafe_allow_html=True)
    st.markdown("### Personalize a new user")
    st.caption("Change the simulated physiology, then see how a few labeled examples shift the model.")
    profile = st.selectbox("Simulated user profile", ["Typical", "Higher resting heart rate", "High skin conductance", "Lower resting temperature", "Custom"], index=1)
    if profile == "Typical":
        shift = np.zeros(4)
    elif profile == "Higher resting heart rate":
        shift = np.array([9.0, 0.0, 0.0, 0.0])
    elif profile == "High skin conductance":
        shift = np.array([0.0, 0.8, 0.0, 0.0])
    elif profile == "Lower resting temperature":
        shift = np.array([0.0, 0.0, -0.5, 0.0])
    else:
        shift = np.array([
            st.slider("Heart rate offset (bpm)", -15, 15, 5),
            st.slider("EDA offset (µS)", -1.0, 1.0, 0.3, 0.05),
            st.slider("Temperature offset (°C)", -1.0, 1.0, -0.2, 0.05),
            st.slider("Motion offset (g)", -0.15, 0.15, 0.03, 0.01),
        ])
    k = st.slider("Calibration windows per class", 1, 20, 3, help="Each calibration window has a known label: baseline or stress.")
    threshold = st.slider("Stress alert threshold", 50, 95, 65, 5) / 100
    if st.button("↻ Generate new session", use_container_width=True):
        st.session_state.seed += 1

seed = st.session_state.seed + int((shift[0] + shift[1] * 10 + shift[2] * 10 + shift[3] * 100) * 100)
x_all, y_all = make_subject(seed, shift)
rng = np.random.default_rng(seed + 9)
indices = np.arange(len(y_all))
support_indices = np.concatenate([rng.choice(indices[y_all == c], size=k, replace=False) for c in (0, 1)])
query_indices = np.setdiff1d(indices, support_indices)
x_support, y_support = x_all[support_indices], y_all[support_indices]
x_query, y_query = x_all[query_indices], y_all[query_indices]
population = base_scores(x_query)
personal = prototype_scores(x_query, x_support, y_support)
pred_pop, pred_personal = population.argmax(axis=1), personal.argmax(axis=1)
acc_pop = (pred_pop == y_query).mean()
acc_personal = (pred_personal == y_query).mean()
gain = acc_personal - acc_pop

st.markdown(f'''<div class="hero"><div><div class="brand">◉ PulseSense</div><h1>Personalized stress detection</h1><p>Few-shot learning adapts a population model to an individual's physiology.</p></div><div class="live">● SIMULATED LIVE SESSION</div></div>''', unsafe_allow_html=True)
st.info("Demo mode: all sensor readings below are simulated. This is a research prototype, not a medical assessment.", icon="ℹ️")

cols = st.columns(4)
metric_card("Population model accuracy", f"{acc_pop:.0%}", "Frozen global baseline")
metric_card("Personalized accuracy", f"{acc_personal:.0%}", f"{2*k} labeled calibration windows", "#68e0ba")
metric_card("Personalization gain", f"{gain:+.0%}", "On this simulated session", "#f4c76c" if gain < 0 else "#68e0ba")
latest_prob = personal[0, 1]
metric_card("Latest stress confidence", f"{latest_prob:.0%}", "Prototype classifier", "#ff987d" if latest_prob >= threshold else "#68e0ba")

if latest_prob >= threshold:
    st.error(f"### Stress detected · {latest_prob:.0%} confidence\n\nThe personalized model's estimated stress probability for the latest simulated window is above your {threshold:.0%} alert threshold.", icon="🔴")
else:
    st.success(f"### No stress detected · {1-latest_prob:.0%} confidence\n\nThe personalized model's estimated stress probability for the latest simulated window is below your {threshold:.0%} alert threshold.", icon="🟢")

left, right = st.columns([1.55, 1])
with left:
    st.markdown('<div class="panel"><div class="eyebrow">01 / Session replay</div><div class="section-title">Personalized predictions over time</div><div class="muted">Stress probability for each held-out window · dashed line is your alert threshold</div></div>', unsafe_allow_html=True)
    chart_n = min(60, len(query_indices))
    show_probs = personal[:chart_n, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, chart_n + 1), y=show_probs, mode="lines", line=dict(color="#55d5b3", width=2), fill="tozeroy", fillcolor="rgba(85,213,179,.12)", name="Personalized"))
    fig.add_trace(go.Scatter(x=np.arange(1, chart_n + 1), y=population[:chart_n, 1], mode="lines", line=dict(color="#8394ee", width=1.5, dash="dot"), name="Population only"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff9a73", annotation_text="alert threshold", annotation_position="top left")
    fig.update_layout(height=310, margin=dict(l=4, r=8, t=12, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,31,.5)", font_color="#91aab5", legend=dict(orientation="h", y=1.15, x=0), xaxis_title="Unseen window", yaxis=dict(range=[0, 1], title="Stress probability", gridcolor="#1c3540"), xaxis=dict(gridcolor="#1c3540"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="panel"><div class="eyebrow">02 / Few-shot adaptation</div><div class="section-title">How the AI adapts</div><div class="muted">A frozen population model plus personal prototypes</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step"><b>1 · Frozen backbone</b><br>Population patterns score the unseen sensor windows.</div><div class="step"><b>2 · User calibration</b><br>{k} labeled baseline + {k} labeled stress windows collected for this person.</div><div class="step"><b>3 · Prototype update</b><br>Average each class in normalized feature space; no retraining required.</div><div class="step"><b>4 · Personalized result</b><br>Classify each new window by distance to the two personal prototypes.</div>', unsafe_allow_html=True)
    st.markdown("**Calibration budget**")
    st.progress(min(1.0, (2 * k) / 40), text=f"{2*k} labeled windows · {min(100, int(100 * (2*k) / 40))}% of 40-window demo budget")
    st.caption(f"Stress alerts: {int((personal[:, 1] >= threshold).sum())} of {len(personal)} unseen windows cross your {threshold:.0%} threshold.")

st.markdown('<div class="panel"><div class="eyebrow">03 / Simulated wearable telemetry</div><div class="section-title">Latest unseen window</div><div class="muted">Four input modalities · values update when you generate a new session</div></div>', unsafe_allow_html=True)
latest = x_query[0]
telemetry = st.columns(4)
for i, c in enumerate(telemetry):
    telemetry[i].metric(FEATURES[i], f"{latest[i]:.1f} {UNITS[i]}", "Simulated sensor reading")

with st.expander("Enter your own readings"):
    st.markdown("Enter a set of wearable readings to get a demo prediction. Use values from the same time window.")
    with st.form("manual_readings"):
        value_cols = st.columns(4)
        hr = value_cols[0].number_input("Heart rate (bpm)", min_value=30.0, max_value=220.0, value=75.0, step=1.0)
        eda = value_cols[1].number_input("EDA (µS)", min_value=0.0, max_value=20.0, value=1.2, step=0.1)
        temp = value_cols[2].number_input("Skin temperature (°C)", min_value=20.0, max_value=40.0, value=33.0, step=0.1)
        motion = value_cols[3].number_input("Motion (g)", min_value=0.0, max_value=5.0, value=0.15, step=0.01)
        submitted = st.form_submit_button("Analyze my readings", use_container_width=True)
    if submitted:
        user_reading = np.array([[hr, eda, temp, motion]])
        user_stress_prob = float(prototype_scores(user_reading, x_support, y_support)[0, 1])
        if user_stress_prob >= threshold:
            st.error(f"### Stress detected · {user_stress_prob:.0%} estimated probability\n\nThis demo model's score is above the selected {threshold:.0%} threshold.", icon="🔴")
        else:
            st.success(f"### No stress detected · {1-user_stress_prob:.0%} estimated probability\n\nThis demo model's score is below the selected {threshold:.0%} threshold.", icon="🟢")
        st.caption("This is an experimental prediction from a toy model and entered values. It is not a medical assessment or a reliable measure of your stress.")

with st.expander("Compare the two model stages"):
    comparison = pd.DataFrame({"Model": ["Population only", "Few-shot personalized"], "Accuracy": [acc_pop, acc_personal], "Calibration labels used": [0, 2*k]})
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("Results are calculated against hidden simulated labels for this session. They are illustrative and do not reproduce the WESAD evaluation metrics in the project writeup.")

st.markdown("<div style='text-align:center;color:#59727e;font-size:11px;padding:22px'>PULSESENSE · INTERACTIVE RESEARCH DEMO · SIMULATED DATA ONLY</div>", unsafe_allow_html=True)
