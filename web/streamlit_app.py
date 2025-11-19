# web/streamlit_app.py
import streamlit as st
import requests
from datetime import datetime, timedelta
from streamlit_lottie import st_lottie
import json
import os

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# CUSTOM CSS + ANIMATIONS
# ------------------------------------------------------
st.markdown("""
<style>
.result-card {
    animation: fadeIn 0.6s ease-out;
    background: rgba(255,255,255,0.95);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    margin-top: 20px;
    color: #212121;  /* Тёмный текст для читаемости */
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.points-anim {
    font-size: 30px;
    font-weight: bold;
    color: #4CAF50;
    animation: bounce 0.7s ease;
    text-align: center;
}
@keyframes bounce {
    0% { transform: scale(0.7); opacity: 0;}
    60% { transform: scale(1.15); opacity: 1;}
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------
def load_lottiefile(filepath: str):
    """Load a Lottie animation safely."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_level(points):
    for pts, name in reversed(LEVELS):
        if points >= pts:
            return len(LEVELS) - LEVELS.index((pts, name)), name
    return 1, "Starter"

def add_points(label):
    game = st.session_state.game
    today = datetime.now().date()
    if game["last_date"] != today:
        yesterday = today - timedelta(days=1)
        game["streak"] = game["streak"] + 1 if game["last_date"] == yesterday else 1
        game["last_date"] = today
    earned = POINTS.get(label, 0)
    game["points"] += earned
    game["stats"][label] = game["stats"].get(label, 0) + 1
    game["level"], _ = get_level(game["points"])
    return earned

# ------------------------------------------------------
# DATA
# ------------------------------------------------------
BINS = {
    "plastic": {"name": "Plastic", "color": "#FFEB3B", "hint": "Rinse bottles and remove caps. Throw into yellow plastic container ♻️", "anim": "animations/plastic.json"},
    "paper": {"name": "Paper", "color": "#2196F3", "hint": "Remove tape/staples. Place in blue paper container 📄", "anim": "animations/paper.json"},
    "glass": {"name": "Glass", "color": "#4CAF50", "hint": "Separate colored and clear glass. Throw into green glass container 🥂", "anim": "animations/glass.json"},
    "metal": {"name": "Metal", "color": "#9E9E9E", "hint": "Crush cans. Use grey metal container 🥫", "anim": "animations/metal.json"},
    "cardboard": {"name": "Cardboard", "color": "#795548", "hint": "Flatten boxes, remove food residue. Brown cardboard container 📦", "anim": "animations/cardboard.json"},
    "trash": {"name": "Mixed Waste", "color": "#212121", "hint": "Only items that cannot be recycled. Black mixed waste container 🗑️", "anim": "animations/trash.json"},
    "rupolice": {"name": "Хуйня позорная", "color": "#FF6F61", "hint": "Этого мусора можете выбросить в BIO отходы!", "anim": "https://gist.github.com/Wiliamins/0bb54b90e95813f0c608788e2cbe343c"}
}

POINTS = {"plastic": 12, "paper": 10, "glass": 15, "metal": 14, "cardboard": 12, "trash": 0}
LEVELS = [(0, "Starter"), (100, "Eco Helper"), (250, "Green Guardian"), (500, "Planet Hero")]

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "game" not in st.session_state:
    st.session_state.game = {"points": 0, "level": 1, "streak": 0, "last_date": None, "stats": {}}
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.title("♻️ EcoVision AI")
st.caption("Smart waste classifier with AI, rewards & sustainability tools.")

# ------------------------------------------------------
# BACKEND URL
# ------------------------------------------------------
backend_url = "https://ecovisionai-1tkk.onrender.com"

# ------------------------------------------------------
# IMAGE UPLOAD & ANALYSIS
# ------------------------------------------------------
uploaded = st.file_uploader("Upload a waste photo", type=["jpg","jpeg","png"])
if uploaded:
    st.image(uploaded, width=260)
    if st.button("Analyze"):
        with st.spinner("Analyzing with AI..."):
            try:
                files = {"file": ("photo.jpg", uploaded.getvalue(), "image/jpeg")}
                resp = requests.post(f"{backend_url}/classify", files=files, timeout=20)
                data = resp.json()
                
                if not data.get("success"):
                    st.error("AI error: " + str(data.get("error")))
                else:
                    label = data["result"]["label"]
                    confidence = data["result"]["confidence"]
                    earned = add_points(label)
                    bin_info = BINS.get(label, BINS["trash"])
                    st.session_state.last_result = {
                        "label": label,
                        "confidence": confidence,
                        "earned": earned,
                        "bin": bin_info,
                        "recommendation": bin_info["hint"]
                    }
            except Exception as e:
                st.error(f"Backend error: {e}")

# ------------------------------------------------------
# RESULT CARD + LOTTIE ANIMATION
# ------------------------------------------------------
if st.session_state.last_result:
    r = st.session_state.last_result
    st.markdown(f"""
    <div class="result-card" style="border-left: 8px solid {r['bin']['color']}">
      <h3 style="text-align:center;color:{r['bin']['color']}">Detected: {r['bin']['name']}</h3>
      <div class="points-anim">+{r['earned']} points</div>
      <p><b>Confidence:</b> {r['confidence']*100:.1f}%</p>
      <p><b>Where to throw:</b> {r['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lottie animation
    anim_data = load_lottiefile(r["bin"]["anim"])
    if anim_data:
        st_lottie(anim_data, height=150)
    
    if st.button("Close"):
        st.session_state.last_result = None
        st.rerun()

# ------------------------------------------------------
# AIR QUALITY
# ------------------------------------------------------
st.header("🌫️ Air Quality Near You")
st.caption("Powered by open environmental data.")

lat = st.number_input("Latitude", value=48.15)
lon = st.number_input("Longitude", value=17.11)

if st.button("Check Air Quality"):
    try:
        resp = requests.get(f"{backend_url}/air_quality?lat={lat}&lon={lon}", timeout=10)
        data = resp.json()
        if data.get("success"):
            st.success(f"AQI: {data['aqi']} — {data['description']}")
        else:
            st.error(f"Air quality error: {data.get('error')}")
    except Exception as e:
        st.error(f"Error: {e}")

# ------------------------------------------------------
# SIDEBAR — GAMIFICATION
# ------------------------------------------------------
game = st.session_state.game
lvl_num, lvl_name = get_level(game["points"])
st.sidebar.header(f"Level {lvl_num}: {lvl_name}")
st.sidebar.metric("Points", game["points"])
st.sidebar.metric("Daily streak", game["streak"])
st.sidebar.write("### Statistics")
for k,v in game["stats"].items():
    st.sidebar.write(f"• {BINS[k]['name']}: {v}")
