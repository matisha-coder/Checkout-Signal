"""
app.py
CheckoutSignal -- a purchase-intent prediction demo styled as a
conversion "signal" dashboard, with a custom traffic-light probability
meter and a live feature-importance breakdown.

Run locally:  streamlit run app.py
Deploy free:  push this folder to a public GitHub repo, then
              go to share.streamlit.io -> New app -> point to app.py
"""

import json
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CheckoutSignal — Purchase Intent Predictor",
    page_icon="🟢",
    layout="wide",
)

# ---------------------------------------------------------
# Design system: fonts, colors, component styling
# ---------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Jost:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg: #0A0E17;
        --surface: #121826;
        --surface-alt: #1A2233;
        --border: #232C40;
        --text: #EAEDF3;
        --muted: #8C96AB;
        --teal: #5EEAD4;
        --green: #34D399;
        --amber: #FBBF24;
        --red: #FB7185;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #0E1420 0%, var(--bg) 45%);
    }

    /* Hide default Streamlit chrome for a cleaner product feel */
    #MainMenu, footer, header {visibility: hidden;}

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: var(--teal);
        margin-bottom: 6px;
    }

    .hero-title {
        font-family: 'Jost', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        letter-spacing: 0.01em;
        margin: 0 0 4px 0;
        color: var(--text);
    }

    .hero-sub {
        color: var(--muted);
        font-size: 0.95rem;
        max-width: 620px;
        line-height: 1.5;
        margin-bottom: 1.6rem;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    .panel-title {
        font-family: 'Jost', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .panel-title .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--teal);
        display: inline-block;
    }

    .panel-sub {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 16px;
    }

    /* Signal meter track */
    .meter-wrap { margin: 10px 0 4px 0; }
    .meter-track {
        position: relative;
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--red) 0%, var(--red) 33%, var(--amber) 33%, var(--amber) 66%, var(--green) 66%, var(--green) 100%);
        opacity: 0.35;
        overflow: visible;
    }
    .meter-marker {
        position: absolute;
        top: -7px;
        width: 4px;
        height: 28px;
        background: #ffffff;
        border-radius: 3px;
        box-shadow: 0 0 10px rgba(255,255,255,0.6);
        transition: left 0.6s cubic-bezier(.22,1,.36,1);
    }
    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--muted);
        margin-top: 8px;
    }

    .verdict-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        margin-top: 4px;
        animation: pulse 2.2s ease-in-out infinite;
    }
    .verdict-green { background: rgba(52,211,153,0.12); color: var(--green); border: 1px solid rgba(52,211,153,0.4); }
    .verdict-amber { background: rgba(251,191,36,0.12); color: var(--amber); border: 1px solid rgba(251,191,36,0.4); }
    .verdict-red   { background: rgba(251,113,133,0.12); color: var(--red); border: 1px solid rgba(251,113,133,0.4); }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(255,255,255,0.08); }
        50%  { box-shadow: 0 0 0 6px rgba(255,255,255,0.0); }
        100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.0); }
    }

    .prob-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1;
        margin: 10px 0 2px 0;
    }

    .factor-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        font-size: 0.82rem;
    }
    .factor-name { width: 190px; color: var(--muted); flex-shrink: 0; }
    .factor-bar-track {
        flex: 1;
        height: 8px;
        background: var(--surface-alt);
        border-radius: 999px;
        overflow: hidden;
    }
    .factor-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--teal), var(--green));
        border-radius: 999px;
    }

    .stat-chip {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .stat-chip .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--teal);
    }
    .stat-chip .lbl {
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 2px;
    }

    /* Streamlit widget restyling */
    .stButton>button {
        background: var(--teal);
        color: #06121A;
        font-family: 'Jost', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #86F0DE;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="eyebrow">ECOMMERCE CONVERSION INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🟢 CheckoutSignal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Reads live session behavior pages browsed, time on site, '
    'bounce and exit patterns and predicts whether the visitor is likely to convert. '
    'Built on a Random Forest classifier trained on real shopper session data.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, encoders, feature_names

@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

model, encoders, feature_names = load_artifacts()
metrics_data = load_metrics()

tab_predict, tab_performance = st.tabs(["🎯 Predict", "📊 Model Performance"])

# ===========================================================
# TAB 1 — PREDICT
# ===========================================================
with tab_predict:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Browsing Activity</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Pages viewed and time spent this session</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            administrative = st.number_input("Administrative pages", 0, 30, 2)
            informational = st.number_input("Informational pages", 0, 30, 1)
            product_related = st.number_input("Product pages", 0, 100, 10)
        with c2:
            product_related_duration = st.number_input("Time on product pages (sec)", 0, 10000, 300)
            page_value = st.number_input("Avg. page value", 0.0, 500.0, 10.0)
            month = st.selectbox(
                "Month",
                ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                index=5,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Engagement Quality</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Signals of how "sticky" the session is</div>', unsafe_allow_html=True)

        bounce_rate = st.slider("Bounce rate", 0.0, 1.0, 0.1)
        exit_rate = st.slider("Exit rate", 0.0, 1.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Visitor Context</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        with vc2:
            weekend = st.checkbox("Weekend session?")
        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("Read the Signal →", use_container_width=True)

    with col_result:
        st.markdown('<div class="panel" style="min-height: 480px;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Conversion Signal</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Live probability read-out for this session</div>', unsafe_allow_html=True)

        if predict_clicked:
            input_dict = {
                "Administrative": administrative,
                "Administrative_Duration": administrative * 30,
                "Informational": informational,
                "Informational_Duration": informational * 30,
                "ProductRelated": product_related,
                "ProductRelated_Duration": product_related_duration,
                "BounceRates": bounce_rate,
                "ExitRates": exit_rate,
                "PageValues": page_value,
                "SpecialDay": 0.0,
                "Month": month,
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": visitor_type,
                "Weekend": weekend,
            }

            row = pd.DataFrame([input_dict])
            for col, le in encoders.items():
                if col in row.columns:
                    row[col] = row[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    row[col] = le.transform(row[col])
            row["Weekend"] = row["Weekend"].astype(int)
            row = row[feature_names]

            prob = float(model.predict_proba(row)[0][1])
            pct = round(prob * 100, 1)

            if prob >= 0.66:
                zone, verdict_class, verdict_text = "green", "verdict-green", "● LIKELY TO CONVERT"
            elif prob >= 0.33:
                zone, verdict_class, verdict_text = "amber", "verdict-amber", "● UNCERTAIN — WATCH"
            else:
                zone, verdict_class, verdict_text = "red", "verdict-red", "● AT RISK OF DROP-OFF"

            color_map = {"green": "#34D399", "amber": "#FBBF24", "red": "#FB7185"}

            st.markdown(f'<div class="prob-number" style="color:{color_map[zone]}">{pct}%</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="verdict-pill {verdict_class}">{verdict_text}</span>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="meter-wrap">
                    <div class="meter-track">
                        <div class="meter-marker" style="left: calc({pct}% - 2px);"></div>
                    </div>
                    <div class="meter-labels">
                        <span>AT RISK</span><span>UNCERTAIN</span><span>CONVERTING</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if metrics_data and metrics_data.get("top_features"):
                st.markdown('<div style="margin-top:26px; color:var(--muted); font-size:0.8rem; margin-bottom:10px;">TOP FACTORS DRIVING THIS MODEL</div>', unsafe_allow_html=True)
                top_feats = metrics_data["top_features"]
                max_imp = max(v for _, v in top_feats)
                for name, imp in top_feats:
                    width = round((imp / max_imp) * 100, 1)
                    st.markdown(
                        f"""
                        <div class="factor-row">
                            <div class="factor-name">{name}</div>
                            <div class="factor-bar-track"><div class="factor-bar-fill" style="width:{width}%;"></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div style="color:var(--muted); font-size:0.88rem; padding-top:20px;">'
                'Fill in the session details and click <b>"Read the Signal"</b> to see the '
                'live conversion probability here.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# TAB 2 — MODEL PERFORMANCE
# ===========================================================
with tab_performance:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Held-Out Test Performance</div>', unsafe_allow_html=True)

    if metrics_data:
        m = metrics_data["metrics"]
        st.markdown('<div class="panel-sub">Random Forest Classifier · evaluated on a 20% held-out split · '
                    f'{m.get("test_size", "—")} sessions</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        stat_items = [
            ("Accuracy", m["accuracy"]), ("Precision", m["precision"]), ("Recall", m["recall"]),
            ("F1 Score", m["f1"]), ("ROC-AUC", m["roc_auc"]),
        ]
        for col, (label, val) in zip(cols, stat_items):
            with col:
                st.markdown(
                    f'<div class="stat-chip"><div class="val">{val}</div><div class="lbl">{label}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="color:var(--muted); font-size:0.88rem;">No metrics.json found yet. '
            'Run <code>python train_model.py</code> first to generate evaluation metrics.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>About This Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:var(--muted); font-size:0.85rem; line-height:1.6;">'
        'Trained on the UCI "Online Shoppers Purchasing Intention" dataset (~12,000 real '
        'browsing sessions). Random Forest was selected over Logistic Regression for its '
        'stronger recall on the minority "purchase" class, which matters more than raw accuracy '
        'in a conversion-prediction setting where missed converters are costlier than false alarms.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; color:var(--muted); font-size:0.72rem; margin-top:30px; '
    'font-family:\'JetBrains Mono\', monospace;">CHECKOUTSIGNAL · RANDOM FOREST · STREAMLIT</div>',
    unsafe_allow_html=True,
)