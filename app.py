import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarTIQ🚗 · Smart Car Intelligence 🚗",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0b0d11;
  --surface:   #13161c;
  --border:    #1f232d;
  --border-hi: #2e3340;
  --accent:    #e8ff47;
  --accent2:   #ff6b35;
  --text:      #d8dce8;
  --muted:     #4a4f60;
  --muted2:    #30343f;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}

/* ── Rich background on the root app wrapper ── */
.stApp {
  background-color: #060810 !important;
  background-image:
    radial-gradient(ellipse 60% 50% at 20% 15%, rgba(232,255,71,0.13) 0%, transparent 55%),
    radial-gradient(ellipse 50% 45% at 82% 78%, rgba(100,180,255,0.10) 0%, transparent 55%),
    radial-gradient(ellipse 40% 35% at 78% 8%,  rgba(255,107,53,0.09) 0%, transparent 50%),
    radial-gradient(ellipse 45% 38% at 8%  88%, rgba(160,100,255,0.08) 0%, transparent 50%) !important;
}

/* ── Animated grid injected via a real div (not pseudo) ── */
#bg-grid {
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(232,255,71,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(232,255,71,0.035) 1px, transparent 1px);
  background-size: 52px 52px;
  pointer-events: none;
  z-index: 0;
  -webkit-mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 20%, transparent 100%);
  mask-image:         radial-gradient(ellipse 80% 80% at 50% 50%, black 20%, transparent 100%);
}

/* ── Floating orbs injected via a real div ── */
#bg-orbs {
  position: fixed;
  inset: 0;
  background:
    radial-gradient(circle 300px at 15% 22%, rgba(232,255,71,0.08) 0%, transparent 65%),
    radial-gradient(circle 250px at 85% 72%, rgba(100,180,255,0.08) 0%, transparent 65%),
    radial-gradient(circle 190px at 72% 10%, rgba(255,107,53,0.07) 0%, transparent 65%),
    radial-gradient(circle 170px at 5%  62%, rgba(160,100,255,0.07) 0%, transparent 65%);
  pointer-events: none;
  z-index: 0;
  animation: orb-drift 14s ease-in-out infinite alternate;
}

@keyframes orb-drift {
  0%   { opacity: 0.6;  transform: scale(1)    translateY(0px)   translateX(0px); }
  33%  { opacity: 0.9;  transform: scale(1.06) translateY(-14px) translateX(6px); }
  66%  { opacity: 0.75; transform: scale(0.97) translateY(8px)   translateX(-5px); }
  100% { opacity: 1.0;  transform: scale(1.03) translateY(-6px)  translateX(8px); }
}

/* ── All real content must sit above bg layers ── */
.main .block-container,
[data-testid="stAppViewContainer"] > section,
[data-testid="stVerticalBlock"] {
  position: relative !important;
  z-index: 2 !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container {
  padding: 2.5rem 1rem 3rem !important;
  max-width: 560px !important;
  position: relative !important;
  z-index: 2 !important;
}

/* ── Price hero card ── */
.price-card {
  background: var(--surface);
  border: 1px solid var(--border-hi);
  border-radius: 20px;
  padding: 2rem 1.8rem 1.6rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.price-card::after {
  content: '';
  position: absolute;
  top: 0; left: 50%; transform: translateX(-50%);
  width: 60%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.pc-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.pc-price {
  font-family: 'Space Mono', monospace;
  font-size: 3.4rem;
  font-weight: 700;
  line-height: 1;
  color: #ffffff;
  letter-spacing: -2px;
}
.pc-price .dollar { color: var(--accent); font-size: 2rem; vertical-align: top; line-height: 1.3; }
.pc-range {
  font-size: 0.74rem;
  color: var(--muted);
  margin-top: 0.55rem;
  font-family: 'Space Mono', monospace;
}
.pc-range b { color: #7a8090; }

/* ── Badge row ── */
.badge-row {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}
.badge {
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 1.5px;
  padding: 0.25rem 0.65rem;
  border-radius: 100px;
  font-weight: 700;
  text-transform: uppercase;
  border: 1px solid;
}
.badge-budget  { color: #4ade80; border-color: rgba(74,222,128,0.3); background: rgba(74,222,128,0.07); }
.badge-mid     { color: #facc15; border-color: rgba(250,204,21,0.3);  background: rgba(250,204,21,0.07);  }
.badge-premium { color: #fb923c; border-color: rgba(251,146,60,0.3);  background: rgba(251,146,60,0.07);  }
.badge-luxury  { color: #c084fc; border-color: rgba(192,132,252,0.3); background: rgba(192,132,252,0.07); }
.badge-body    { color: var(--accent); border-color: rgba(232,255,71,0.25); background: rgba(232,255,71,0.05); }
.badge-drive   { color: #67e8f9; border-color: rgba(103,232,249,0.25); background: rgba(103,232,249,0.05); }

/* ── Stat strip ── */
.stat-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem;
  margin-top: 1rem;
}
.stat-cell {
  background: var(--muted2);
  border-radius: 10px;
  padding: 0.6rem 0.4rem;
  text-align: center;
}
.sc-val {
  font-family: 'Space Mono', monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--accent);
}
.sc-lbl {
  font-size: 0.57rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 0.2rem;
}

/* ── Config card ── */
.config-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.4rem 1.6rem 1.6rem;
  margin-bottom: 1.2rem;
}
.cc-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 1.1rem;
  padding-bottom: 0.7rem;
  border-bottom: 1px solid var(--border);
}

/* ── Form overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
  background: var(--muted2) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
}
label {
  color: var(--muted) !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  font-family: 'Space Mono', monospace !important;
}

/* Accent on focus */
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
  border-color: rgba(232,255,71,0.4) !important;
  box-shadow: 0 0 0 3px rgba(232,255,71,0.08) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
  background: var(--border-hi) !important;
}

/* ── CTA Button ── */
.stButton > button {
  background: var(--accent) !important;
  color: #0b0d11 !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 12px !important;
  width: 100% !important;
  padding: 0.85rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 24px rgba(232,255,71,0.2) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 36px rgba(232,255,71,0.35) !important;
}

/* ── Empty state ── */
.empty-card {
  background: var(--surface);
  border: 1px dashed var(--border-hi);
  border-radius: 20px;
  padding: 2.8rem 1.5rem;
  text-align: center;
  margin-bottom: 1.2rem;
}
.ec-icon { font-size: 2.8rem; opacity: 0.3; margin-bottom: 0.7rem; }
.ec-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  letter-spacing: 3px;
  color: var(--muted);
  text-transform: uppercase;
}
.ec-sub { font-size: 0.76rem; color: var(--muted2); margin-top: 0.4rem; font-weight: 300; }

/* ── Divider ── */
.hdiv { border: none; border-top: 1px solid var(--border); margin: 1.1rem 0; }

/* ── Two-col grid ── */
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 1rem;
}

/* ── Header ── */
.app-header { text-align: center; padding: 0 0 1.6rem; }
.ah-eyebrow {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.35rem;
}
.ah-title {
  font-family: 'Space Mono', monospace;
  font-size: 1.9rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.5px;
}
.ah-title span { color: var(--accent); }
.ah-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; font-weight: 300; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    ridge  = joblib.load('ridge_poly_regression_model.pkl')
    scaler = joblib.load('standard_scaler_poly.pkl')
    num_cols = ['symboling','wheelbase','carlength','carwidth',
                'curbweight','enginesize','horsepower','citympg']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(pd.DataFrame(0, index=[0], columns=num_cols))
    return ridge, scaler, poly, num_cols

try:
    ridge_model, scaler_poly, poly_tf, numerical_cols = load_artifacts()
    poly_feat_names = list(poly_tf.get_feature_names_out(numerical_cols))
    model_loaded = True
except Exception:
    model_loaded = False

CAT_DUMMY_COLS = [
    'carbody_sedan','drivewheel_fwd','drivewheel_rwd','enginelocation_rear',
    'enginetype_dohcv','enginetype_l','enginetype_ohc','enginetype_ohcf',
    'enginetype_ohcv','enginetype_rotor','cylindernumber_five',
    'cylindernumber_four','cylindernumber_six'
]

# ─── Header ───────────────────────────────────────────────────────────────────
# Inject background divs — safe alternative to ::before/::after pseudo-elements
st.markdown("""
<div id="bg-grid"></div>
<div id="bg-orbs"></div>
<div class="app-header">
  <div class="ah-eyebrow">Smart Car Intelligence · Ridge · Poly</div>
  <div class="ah-title">Car<span>TIQ</span></div>
  <div class="ah-sub">IQ-powered price estimation for any car</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Place `ridge_poly_regression_model.pkl` and `standard_scaler_poly.pkl` alongside `app.py`.")
    st.stop()

# ─── Result placeholder ───────────────────────────────────────────────────────
result_slot = st.empty()

# ─── Config Card ──────────────────────────────────────────────────────────────
st.markdown('<div class="config-card"><div class="cc-title">// Vehicle Configuration</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    carbody     = st.selectbox("Body Style",        ['sedan','hatchback','wagon','hardtop','convertible'])
    drivewheel  = st.selectbox("Drive",             ['fwd','rwd','4wd'])
    enginetype  = st.selectbox("Engine Type",       ['ohc','dohc','dohcv','ohcv','ohcf','l','rotor'])
    cylindernumber = st.selectbox("Cylinders",      ['four','six','five','eight','two','three','twelve'])
    enginelocation = st.selectbox("Engine Location",['front','rear'])

with c2:
    enginesize  = st.number_input("Engine Size (cc)",  60,  330, 126, step=1)
    horsepower  = st.number_input("Horsepower (hp)",   40,  300, 104, step=1)
    citympg     = st.number_input("City MPG",          10,   55,  25, step=1)
    curbweight  = st.number_input("Curb Weight (lbs)", 1400,4100,2555, step=10)

st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    wheelbase = st.number_input("Wheelbase (in)",  80.0, 125.0,  98.7, step=0.1, format="%.1f")
    carlength = st.number_input("Length (in)",    140.0, 210.0, 174.0, step=0.1, format="%.1f")
with c4:
    carwidth  = st.number_input("Width (in)",      60.0,  75.0,  65.9, step=0.1, format="%.1f")
    symboling = st.slider("Risk Rating", -3, 3, 0, help="-3 safest · +3 sporty")

st.markdown('</div>', unsafe_allow_html=True)

# ─── CTA ─────────────────────────────────────────────────────────────────────
predict = st.button("ESTIMATE PRICE →")

# ─── Result ───────────────────────────────────────────────────────────────────
with result_slot.container():
    if predict:
        row = {c: 0.0 for c in CAT_DUMMY_COLS}
        row.update({'symboling':symboling,'wheelbase':wheelbase,'carlength':carlength,
                    'carwidth':carwidth,'curbweight':curbweight,'enginesize':enginesize,
                    'horsepower':horsepower,'citympg':citympg})
        df = pd.DataFrame([row])

        if carbody == 'sedan':       df['carbody_sedan'] = 1.0
        if drivewheel == 'fwd':      df['drivewheel_fwd'] = 1.0
        elif drivewheel == 'rwd':    df['drivewheel_rwd'] = 1.0
        if enginelocation == 'rear': df['enginelocation_rear'] = 1.0
        et_map = {'dohcv':'enginetype_dohcv','l':'enginetype_l','ohc':'enginetype_ohc',
                  'ohcf':'enginetype_ohcf','ohcv':'enginetype_ohcv','rotor':'enginetype_rotor'}
        if enginetype in et_map:     df[et_map[enginetype]] = 1.0
        cn_map = {'five':'cylindernumber_five','four':'cylindernumber_four','six':'cylindernumber_six'}
        if cylindernumber in cn_map: df[cn_map[cylindernumber]] = 1.0

        poly_num = poly_tf.transform(df[numerical_cols])
        poly_df  = pd.DataFrame(poly_num, columns=poly_feat_names, index=df.index)
        final    = pd.concat([poly_df, df[CAT_DUMMY_COLS]], axis=1)
        final    = final[poly_feat_names + CAT_DUMMY_COLS]

        price  = max(ridge_model.predict(scaler_poly.transform(final))[0], 0)
        lo, hi = price * 0.92, price * 1.08
        hp_k   = round(horsepower / (price / 1000), 2) if price > 0 else 0

        if price < 10000:    tier, tbadge = "Budget",    "badge-budget"
        elif price < 20000:  tier, tbadge = "Mid-Range", "badge-mid"
        elif price < 35000:  tier, tbadge = "Premium",   "badge-premium"
        else:                tier, tbadge = "Luxury",    "badge-luxury"

        # MPG badge
        if citympg >= 35:    mpg_c, mpg_l = "#4ade80", "Efficient"
        elif citympg >= 25:  mpg_c, mpg_l = "#facc15", "Moderate"
        else:                mpg_c, mpg_l = "#f87171",  "Thirsty"

        st.markdown(f"""
        <div class="price-card">
          <div class="pc-label">Estimated Market Value</div>
          <div class="pc-price"><span class="dollar">$</span>{price:,.0f}</div>
          <div class="pc-range">Range &nbsp;<b>${lo:,.0f}</b> — <b>${hi:,.0f}</b>&nbsp; (± 8%)</div>

          <div class="badge-row">
            <span class="badge {tbadge}">{tier}</span>
            <span class="badge badge-body">{carbody.title()}</span>
            <span class="badge badge-drive">{drivewheel.upper()}</span>
            <span class="badge" style="color:{mpg_c};border-color:{mpg_c}44;background:{mpg_c}0e">{mpg_l}</span>
          </div>

          <div class="stat-strip">
            <div class="stat-cell">
              <div class="sc-val">{horsepower}</div>
              <div class="sc-lbl">HP</div>
            </div>
            <div class="stat-cell">
              <div class="sc-val">{hp_k}</div>
              <div class="sc-lbl">HP/$1K</div>
            </div>
            <div class="stat-cell">
              <div class="sc-val">{citympg}</div>
              <div class="sc-lbl">City MPG</div>
            </div>
            <div class="stat-cell">
              <div class="sc-val">{symboling:+d}</div>
              <div class="sc-lbl">Risk</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-card">
          <div class="ec-icon">🚗</div>
          <div class="ec-title">Awaiting Config</div>
          <div class="ec-sub">Fill specs below · Hit estimate</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.58rem;
            color:#1f232d;letter-spacing:3px;text-transform:uppercase;padding-top:1rem">
  CarTIQ · Smart Car Intelligence · Ridge Poly · scikit-learn
</div>
""", unsafe_allow_html=True)



