import streamlit as st
import pandas as pd
import joblib
import re
import difflib
import plotly.express as px
from typing import Optional, Union

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EV Chatbot Pro",
    page_icon="⚡",
    layout="wide",
)

# -------------------------------------------------
# CUSTOM PREMIUM CSS
# -------------------------------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(200,200,200,0.4);
    padding: 20px;
}

.sidebar-title {
    font-size: 30px;
    font-weight: 900;
    margin-top: 20px;
    margin-bottom: 20px;
}

.profile-card {
    padding: 14px;
    background: rgba(255,255,255,0.5);
    border-radius: 16px;
    border: 1px solid #e3e3e3;
    text-align: center;
    margin-bottom: 20px;
}
.avatar {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: linear-gradient(135deg,#2563eb,#4f83ff);
    color:white;
    font-size:30px;
    font-weight:700;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:auto;
}

.stButton button {
    border-radius: 12px !important;
    padding: 8px 12px !important;
    font-weight: 600;
}

.card {
    padding: 22px;
    background: rgba(255,255,255,0.6);
    border-radius: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.1);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SESSION STATE (page + chat)
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
with st.sidebar:
    st.markdown(
        """
    <div class='profile-card'>
        <div class='avatar'>P</div>
        <h4 style='margin-top:10px;'>Prakhar Kumar/h4>
        <p style='font-size:13px;color:#555;'>EV Dashboard User</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-title'>⚡ EV Assistant</div>", unsafe_allow_html=True)

    if st.button("💬  Chatbot"):
        st.session_state.page = "Chatbot"
        st.rerun()

    if st.button("📊  Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button("📈  Analytics"):
        st.session_state.page = "Analytics"
        st.rerun()

    if st.button("ℹ  About"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown(
        """
    <div class='profile-card'>
        <h4>🔋 EV Intelligence Suite</h4>
        <p style='font-size:13px;'>Chatbot • Analytics • Predictions • Insights</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# LOAD MODEL + DATA
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("linear_regression_model.pkl")
    except Exception:
        return None


@st.cache_data
def load_data():
    df = pd.read_csv("electric_vehicles_spec_2025.csv")
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.lower()
    )
    return df

model = load_model()
data = load_data()

# -------------------------------------------------
# BUILD FULL PREDICTION ROW (THE FIX)
# -------------------------------------------------
def build_prediction_row(df: pd.DataFrame, battery, range_km):

    row = {}

    for col in df.columns:
        if col == "battery_capacity_kwh":
            row[col] = battery
        elif col == "range_km":
            row[col] = range_km
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                row[col] = df[col].median()
            else:
                row[col] = df[col].mode()[0]

    return pd.DataFrame([row])

# -------------------------------------------------
# QUERY PARSER
# -------------------------------------------------
def extract_query_details(text: str, df: pd.DataFrame):

    text_low = text.lower()

    details: dict[str, Optional[Union[float, int, str]]] = {
        "battery": None,
        "range": None,
        "budget": None,
        "model": None,
        "brand": None,
    }

    m = re.search(r"(\d+)\s*(kwh|kw|battery)", text_low)
    if m:
        details["battery"] = float(m.group(1))

    m = re.search(r"(\d+)\s*(km|range)", text_low)
    if m:
        details["range"] = float(m.group(1))

    m = re.search(r"(\d+)\s*(lakh|lakhs|million|cr|crore)", text_low)
    if m:
        n = int(m.group(1))
        u = m.group(2)
        if "lakh" in u:
            details["budget"] = n * 100000
        elif u in ["cr", "crore"]:
            details["budget"] = n * 10000000
        else:
            details["budget"] = n * 1000000

    # fuzzy brand+model detection
    combined = (df["brand"] + " " + df["model"]).str.lower().tolist()
    match = difflib.get_close_matches(text_low, combined, n=1, cutoff=0.25)
    if match:
        idx = combined.index(match[0])
        details["brand"] = str(df.iloc[idx]["brand"])
        details["model"] = str(df.iloc[idx]["model"])

    return details

# -------------------------------------------------
# INTENT DETECTOR
# -------------------------------------------------
def detect_intent(text: str) -> str:
    t = text.lower()
    if "price" in t or "estimate" in t:
        return "price"
    if "under" in t or "budget" in t:
        return "budget"
    if "recommend" in t or "suggest" in t:
        return "recommend"
    if any(k in t for k in ["info", "details", "specs", "tell me about"]):
        return "info"
    return "unknown"

# -------------------------------------------------
# CHATBOT ENGINE
# -------------------------------------------------
def chatbot_reply(q: str) -> str:
    intent = detect_intent(q)
    d = extract_query_details(q, data)

    # PRICE
    if intent == "price":
        if model is None:
            return "Price prediction unavailable (model file missing)."

        if not d["battery"] or not d["range"]:
            return "Please provide both **battery size (kWh)** and **range (km)**."

        try:
            # Build full row automatically
            df_clean = data.drop(columns=["price_inr", "source_url"], errors="ignore")
            pred_input = build_prediction_row(df_clean, d["battery"], d["range"])
            price = model.predict(pred_input)[0]
            return f"Estimated Price: **₹{price:,.0f}**"
        except Exception as e:
            return f"Prediction failed: {e}"

    # BUDGET
    if intent == "budget":
        if not d["budget"]:
            return "Please provide a budget like **under 15 lakh**."
        df_b = data[data["price_inr"] <= d["budget"]]
        if df_b.empty:
            return "No EVs match this budget."
        return "\n".join(
            [f"- {r.brand} {r.model} ({r.range_km} km)" for _, r in df_b.head(5).iterrows()]
        )

    # RECOMMEND
    if intent == "recommend":
        top = data.sort_values("range_km", ascending=False).head(5)
        return "\n".join(
            [f"- {r.brand} {r.model} – {r.range_km} km" for _, r in top.iterrows()]
        )

    # INFO
    if intent == "info":
        if not d["model"]:
            return "I couldn't identify a specific EV model."
        row = data[(data["brand"] == d["brand"]) & (data["model"] == d["model"])]
        if row.empty:
            return "Model not found in dataset."
        r = row.iloc[0]
        return (
            f"### 🚗 {r.brand} {r.model}\n"
            f"- Battery: {r.get('battery_capacity_kwh','N/A')} kWh\n"
            f"- Range: {r.get('range_km','N/A')} km\n"
            f"- Body: {r.get('body_style','N/A')}\n"
            f"- Charging: {r.get('charging_type','N/A')}"
        )

    return "I didn't understand. Try asking about **price**, **budget**, **recommendations**, or **EV details**."

# -------------------------------------------------
# PAGE HANDLER
# -------------------------------------------------
page = st.session_state.page

# CHATBOT PAGE
if page == "Chatbot":
    st.title("🤖 EV Chatbot")

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).markdown(msg["message"])

    q = st.chat_input("Ask about EVs...")
    if q:
        st.session_state.chat.append({"role": "user", "message": q})
        reply = chatbot_reply(q)
        st.session_state.chat.append({"role": "assistant", "message": reply})
        st.rerun()

# DASHBOARD
elif page == "Dashboard":
    st.title("📊 EV Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Battery", f"{data['battery_capacity_kwh'].mean():.1f} kWh")
    col2.metric("Avg Range", f"{data['range_km'].mean():.0f} km")
    col3.metric("Total Models", data.shape[0])

    st.subheader("Battery Distribution")
    st.plotly_chart(px.histogram(data, x="battery_capacity_kwh"), use_container_width=True)

    st.subheader("Top Brands")
    bc = data["brand"].value_counts().reset_index()
    bc.columns = ["brand", "count"]
    st.plotly_chart(px.bar(bc, x="brand", y="count"), use_container_width=True)

# ANALYTICS
elif page == "Analytics":
    st.title("📈 EV Analytics")

    st.subheader("Battery vs Range")
    st.plotly_chart(
        px.scatter(
            data,
            x="battery_capacity_kwh",
            y="range_km",
            color="brand",
            size="range_km",
            hover_data=["model"],
        ),
        use_container_width=True,
    )

    st.subheader("Brand-wise Avg Range")
    avg = data.groupby("brand")["range_km"].mean().reset_index()
    st.plotly_chart(px.bar(avg, x="brand", y="range_km"), use_container_width=True)

# ABOUT
elif page == "About":
    st.title("ℹ About EV Chatbot Pro")
    st.write(
        """
**EV Chatbot Pro** includes:
- Smart EV Chatbot  
- Price Estimation  
- Model Details  
- Dashboard + Analytics  
"""
    )


