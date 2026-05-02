import streamlit as st
import numpy as np
import pickle
import os
import io
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(
    page_title="PneumoScan | PUMA-CNN",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Default: LIGHT mode ───────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

dark = st.session_state.dark_mode

# ── Theme tokens ──────────────────────────────────────────────────
if dark:
    bg_main          = "#060b14"
    bg_sidebar       = "linear-gradient(180deg,#040810 0%,#070e1c 100%)"
    bg_card          = "linear-gradient(135deg,#070d1c,#0b1628)"
    sidebar_text     = "#6aaed0"
    hero_h1          = "#e8f4ff"
    hero_p           = "#7aaccc"
    section_lbl      = "#00c8f077"
    stat_num         = "#00c8f0"
    stat_lbl         = "#4a7a9a"
    body_text        = "#7aaccc"
    muted_text       = "#3a6a8a"
    bdr              = "#0c2236"
    bdr_top          = "#00c8f033"
    input_bg         = "#040810"
    input_bdr        = "#0c1e30"
    input_color      = "#6aaed0"
    card_bdr         = "#0c1e30"
    about_bdr        = "#00c8f055"
    about_h4         = "#00c8f0bb"
    about_p          = "#6aaed0"
    ref_key          = "#00c8f077"
    ref_val          = "#4a7a9a"
    footer_color     = "#1e3e58"
    warn_bg          = "linear-gradient(135deg,#160e00,#201400)"
    warn_bdr_l       = "#f0a000"
    warn_bdr         = "#604000"
    warn_h4          = "#f0a000"
    warn_p           = "#9a7030"
    hr_color         = "#0c1e30"
    pneu_bg          = "linear-gradient(145deg,#120202,#1e0606)"
    pneu_bdr         = "#5a0808"
    norm_bg          = "linear-gradient(145deg,#020f06,#031508)"
    norm_bdr         = "#084018"
    unc_bg           = "linear-gradient(145deg,#0c0c02,#141000)"
    unc_bdr          = "#4a3e04"
    rec_pneu_bg      = "#120202"
    rec_norm_bg      = "#020f06"
    rec_unc_bg       = "#0c0a02"
    pill_bg          = "#0a1628"
    pill_bdr         = "#00c8f033"
    pill_color       = "#00c8f0"
    pill_key         = "#4a7a9a"
    toggle_icon      = "☀️"
    toggle_label     = "Light Mode"
    prob_bar_bg      = "#040810"
    hero_accent      = "#00c8f0"
    badge_bg         = "rgba(0,200,240,0.08)"
    badge_bdr        = "rgba(0,200,240,0.2)"
    sidebar_pill     = "#070e1c"
    sidebar_pill_bdr = "#0c2236"
    puma_node_bg     = "#0a1628"
    puma_node_bdr    = "#00c8f044"
    puma_node_text   = "#00c8f0"
    puma_phase_bg    = "#050c1a"
    puma_phase_bdr   = "#0c2236"
    puma_phase_text  = "#6aaed0"
    puma_arrow       = "#00c8f033"
    puma_tag_bg      = "rgba(0,200,240,0.1)"
    puma_tag_color   = "#00c8f0"
    puma_tag_bdr     = "#00c8f033"
    xray_wrap_bg     = "#020508"
else:
    bg_main          = "#f0f5fa"
    bg_sidebar       = "linear-gradient(180deg,#dce8f5 0%,#e8f2fb 100%)"
    bg_card          = "linear-gradient(135deg,#ffffff,#eaf3fc)"
    sidebar_text     = "#1a3555"
    hero_h1          = "#0a2240"
    hero_p           = "#2a4870"
    section_lbl      = "#006088"
    stat_num         = "#004d77"
    stat_lbl         = "#2a5070"
    body_text        = "#1a3555"
    muted_text       = "#2a4870"
    bdr              = "#aac8e0"
    bdr_top          = "#0088bb"
    input_bg         = "#ffffff"
    input_bdr        = "#aac8e0"
    input_color      = "#0a2240"
    card_bdr         = "#aac8e0"
    about_bdr        = "#0088bb"
    about_h4         = "#004d77"
    about_p          = "#1a3555"
    ref_key          = "#0066aa"
    ref_val          = "#2a4870"
    footer_color     = "#5a7898"
    warn_bg          = "linear-gradient(135deg,#fff8e0,#fff2c8)"
    warn_bdr_l       = "#d4980a"
    warn_bdr         = "#b88020"
    warn_h4          = "#8c6200"
    warn_p           = "#5a4008"
    hr_color         = "#aac8e0"
    pneu_bg          = "linear-gradient(145deg,#fff0f0,#ffe5e5)"
    pneu_bdr         = "#d07070"
    norm_bg          = "linear-gradient(145deg,#f0fff4,#e4ffe8)"
    norm_bdr         = "#40a860"
    unc_bg           = "linear-gradient(145deg,#fffde5,#fff7d0)"
    unc_bdr          = "#b89010"
    rec_pneu_bg      = "#fff0f0"
    rec_norm_bg      = "#f0fff4"
    rec_unc_bg       = "#fffde5"
    pill_bg          = "#eaf3fc"
    pill_bdr         = "#0088bb"
    pill_color       = "#004d77"
    pill_key         = "#2a5070"
    toggle_icon      = "🌙"
    toggle_label     = "Dark Mode"
    prob_bar_bg      = "#c8dce8"
    hero_accent      = "#0088bb"
    badge_bg         = "rgba(0,136,187,0.08)"
    badge_bdr        = "rgba(0,136,187,0.25)"
    sidebar_pill     = "#eaf3fc"
    sidebar_pill_bdr = "#aac8e0"
    puma_node_bg     = "#eaf3fc"
    puma_node_bdr    = "#0088bb"
    puma_node_text   = "#004d77"
    puma_phase_bg    = "#f5faff"
    puma_phase_bdr   = "#aac8e0"
    puma_phase_text  = "#1a3555"
    puma_arrow       = "#aac8e0"
    puma_tag_bg      = "rgba(0,136,187,0.08)"
    puma_tag_color   = "#004d77"
    puma_tag_bdr     = "#0088bb"
    xray_wrap_bg     = "#1a1a1a"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,700;0,9..144,900;1,9..144,400&display=swap');
*,*::before,*::after{{box-sizing:border-box;}}
html,body,[class*="css"]{{
    font-family:'DM Sans',sans-serif;
    background:{bg_main} !important;
    color:{body_text};
}}
section[data-testid="stSidebar"]{{
    background:{bg_sidebar} !important;
    border-right:1px solid {bdr};
    width:230px !important;
}}
section[data-testid="stSidebar"] *{{color:{sidebar_text} !important;}}
section[data-testid="stSidebar"] .stRadio > label{{display:none !important;}}
section[data-testid="stSidebar"] .stRadio > div{{
    gap:6px !important;display:flex !important;
    flex-direction:column !important;width:100% !important;
}}
section[data-testid="stSidebar"] .stRadio > div > label{{
    display:flex !important;align-items:center !important;
    padding:13px 16px !important;border-radius:10px !important;
    font-size:0.88rem !important;font-weight:600 !important;
    background:{sidebar_pill} !important;border:1px solid {sidebar_pill_bdr} !important;
    cursor:pointer !important;min-height:48px !important;
    width:100% !important;margin:0 !important;transition:border-color 0.18s ease !important;
}}
section[data-testid="stSidebar"] .stRadio > div > label:hover{{
    border-color:{hero_accent} !important;background:{badge_bg} !important;
}}
.main .block-container{{max-width:1200px;padding:2rem 2.5rem 4rem 2.5rem !important;}}
.brand-wrap{{padding:1.8rem 0 1.4rem 0;}}
.brand-icon{{
    width:40px;height:40px;
    background:linear-gradient(135deg,{hero_accent}22,{hero_accent}44);
    border:1px solid {hero_accent}55;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.25rem;margin-bottom:10px;
}}
.brand-name{{
    font-family:'Fraunces',serif;font-size:1.35rem;
    color:{stat_num};font-weight:700;letter-spacing:-0.3px;line-height:1.1;
}}
.brand-sub{{
    font-size:0.58rem;color:{muted_text};letter-spacing:3px;
    text-transform:uppercase;margin-top:4px;font-weight:600;
}}
.page-hero{{padding:3rem 0 0.5rem 0;margin-bottom:2.5rem;}}
.hero-badge{{
    display:inline-flex;align-items:center;gap:6px;
    background:{badge_bg};border:1px solid {badge_bdr};
    border-radius:100px;padding:5px 14px;
    font-size:0.68rem;font-weight:700;color:{hero_accent};
    text-transform:uppercase;letter-spacing:2px;margin-bottom:1.2rem;
}}
.hero-dot{{
    width:6px;height:6px;background:{hero_accent};
    border-radius:50%;animation:pulse 2s infinite;
}}
@keyframes pulse{{
    0%,100%{{opacity:1;transform:scale(1);}}
    50%{{opacity:0.4;transform:scale(0.7);}}
}}
.page-hero h1{{
    font-family:'Fraunces',serif;font-size:2.8rem;font-weight:900;
    color:{hero_h1};margin:0 0 0.6rem 0;line-height:1.1;letter-spacing:-1px;
}}
.page-hero h1 em{{font-style:italic;color:{hero_accent};}}
.page-hero p{{
    color:{hero_p};font-size:0.95rem;margin:0;
    font-weight:400;max-width:560px;line-height:1.6;
}}
.hero-rule{{
    height:1px;background:linear-gradient(90deg,{hero_accent}33,transparent);
    margin-top:2.2rem;border:none;
}}
.section-label{{
    font-size:0.6rem;letter-spacing:3.5px;text-transform:uppercase;
    color:{section_lbl};margin-bottom:0.75rem;font-weight:700;
    display:flex;align-items:center;gap:8px;
}}
.section-label::before{{
    content:'';display:inline-block;width:16px;height:1px;
    background:{hero_accent};opacity:0.5;
}}
.stat-block{{
    background:{bg_card};border:1px solid {bdr};border-top:2px solid {bdr_top};
    border-radius:12px;padding:1.6rem 1.2rem 1.4rem;text-align:center;
    position:relative;overflow:hidden;
}}
.stat-block::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:60px;
    background:linear-gradient(180deg,{hero_accent}08,transparent);pointer-events:none;
}}
.stat-num{{
    font-family:'Fraunces',serif;font-size:2.3rem;color:{stat_num};
    line-height:1;margin-bottom:7px;font-weight:700;letter-spacing:-1px;
}}
.stat-label{{font-size:0.62rem;color:{stat_lbl};text-transform:uppercase;letter-spacing:2.5px;font-weight:600;}}
.result-card{{
    border-radius:16px;padding:2.2rem 1.8rem 1.8rem;
    text-align:center;margin-bottom:1.4rem;position:relative;overflow:hidden;
}}
.result-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.result-card.pneumonia{{background:{pneu_bg};border:1px solid {pneu_bdr};}}
.result-card.pneumonia::before{{background:linear-gradient(90deg,#cc2222,#ff4444);}}
.result-card.normal{{background:{norm_bg};border:1px solid {norm_bdr};}}
.result-card.normal::before{{background:linear-gradient(90deg,#117733,#22cc55);}}
.result-card.uncertain{{background:{unc_bg};border:1px solid {unc_bdr};}}
.result-card.uncertain::before{{background:linear-gradient(90deg,#aa7700,#ddaa00);}}
.result-card .icon{{font-size:2rem;margin-bottom:0.4rem;display:block;line-height:1;}}
.result-card .verdict{{
    font-family:'Fraunces',serif;font-size:1.7rem;font-weight:700;
    margin:0.3rem 0;letter-spacing:-0.5px;
}}
.result-card.pneumonia .verdict{{color:#dd2222;}}
.result-card.normal    .verdict{{color:#117733;}}
.result-card.uncertain .verdict{{color:#aa7700;}}
.result-card .score{{
    font-family:'Fraunces',serif;font-size:3.5rem;font-weight:900;
    line-height:1;margin:0.8rem 0 0.3rem;letter-spacing:-2px;
}}
.result-card.pneumonia .score{{color:#dd2222;}}
.result-card.normal    .score{{color:#117733;}}
.result-card.uncertain .score{{color:#aa7700;}}
.result-card .sublabel{{
    font-size:0.62rem;color:{muted_text};text-transform:uppercase;letter-spacing:2.5px;font-weight:600;
}}
.result-card .divider{{height:1px;background:{bdr};margin:1.2rem 0;}}
.prob-section{{margin-top:0.4rem;}}
.prob-row{{margin:12px 0;}}
.prob-label{{
    display:flex;justify-content:space-between;align-items:center;
    font-size:0.83rem;color:{body_text};margin-bottom:7px;font-weight:600;
}}
.prob-pct{{font-family:'Fraunces',serif;font-size:1rem;font-weight:700;letter-spacing:-0.5px;}}
.prob-bar-bg{{
    background:{prob_bar_bg};border-radius:100px;height:8px;
    width:100%;overflow:hidden;border:1px solid {bdr};
}}
.prob-bar-fill-p{{background:linear-gradient(90deg,#991111,#ee3333);height:100%;border-radius:100px;}}
.prob-bar-fill-n{{background:linear-gradient(90deg,#0d4422,#1a9944);height:100%;border-radius:100px;}}
.rec-panel{{border-radius:14px;padding:1.6rem 1.8rem;margin-top:1.2rem;}}
.rec-panel.pneumonia{{background:{rec_pneu_bg};border:1px solid {pneu_bdr};border-left:4px solid #cc2222;}}
.rec-panel.normal{{background:{rec_norm_bg};border:1px solid {norm_bdr};border-left:4px solid #1a9944;}}
.rec-panel.uncertain{{background:{rec_unc_bg};border:1px solid {unc_bdr};border-left:4px solid #cc9900;}}
.rec-panel h4{{
    font-size:0.78rem;text-transform:uppercase;letter-spacing:2px;
    margin:0 0 1rem 0;font-weight:700;
}}
.rec-panel.pneumonia h4{{color:#cc2222;}}
.rec-panel.normal    h4{{color:#117733;}}
.rec-panel.uncertain h4{{color:#aa7700;}}
.rec-item{{
    display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;
    font-size:0.85rem;line-height:1.65;color:{body_text};font-weight:400;
}}
.rec-dot{{width:6px;height:6px;border-radius:50%;margin-top:8px;flex-shrink:0;}}
.rec-panel.pneumonia .rec-dot{{background:#cc2222;}}
.rec-panel.normal    .rec-dot{{background:#22aa55;}}
.rec-panel.uncertain .rec-dot{{background:#cc9900;}}
.warning-banner{{
    background:{warn_bg};border:1px solid {warn_bdr};
    border-left:4px solid {warn_bdr_l};border-radius:12px;
    padding:1.4rem 1.8rem;margin-bottom:1.8rem;
}}
.warning-banner h4{{
    color:{warn_h4};font-size:0.82rem;text-transform:uppercase;
    letter-spacing:2px;margin:0 0 0.6rem 0;font-weight:700;
}}
.warning-banner p{{color:{warn_p};font-size:0.86rem;line-height:1.7;margin:0;font-weight:400;}}
.puma-card{{
    background:{bg_card};border:1px solid {card_bdr};
    border-radius:12px;padding:1.5rem 1.6rem;margin-bottom:1rem;
    transition:border-color 0.2s;
}}
.puma-card:hover{{border-color:{hero_accent}44;}}
.puma-card h4{{
    color:{stat_num};font-size:0.83rem;font-weight:700;
    margin:0 0 0.6rem 0;text-transform:uppercase;letter-spacing:1.5px;
    display:flex;align-items:center;gap:8px;
}}
.puma-card h4::before{{
    content:'';width:3px;height:14px;background:{hero_accent};
    border-radius:2px;display:inline-block;flex-shrink:0;
}}
.puma-card p{{color:{body_text};font-size:0.86rem;line-height:1.75;margin:0;font-weight:400;}}
.hp-grid{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:2rem;}}
.hp-pill{{
    display:flex;flex-direction:column;align-items:flex-start;
    background:{pill_bg};border:1px solid {pill_bdr};
    border-radius:10px;padding:10px 16px 12px;min-width:120px;
}}
.hp-pill-key{{
    font-size:0.58rem;color:{pill_key};text-transform:uppercase;
    letter-spacing:2px;font-weight:700;margin-bottom:5px;white-space:nowrap;
}}
.hp-pill-val{{
    font-family:'Courier New',monospace;font-size:0.92rem;
    color:{pill_color};font-weight:700;
}}
.about-block{{
    background:{bg_card};border:1px solid {card_bdr};
    border-left:3px solid {about_bdr};border-radius:0 12px 12px 0;
    padding:1.4rem 1.6rem;margin-bottom:1rem;
}}
.about-block h4{{color:{about_h4};font-size:0.88rem;margin:0 0 0.5rem 0;font-weight:700;}}
.about-block p{{color:{about_p};font-size:0.85rem;line-height:1.75;margin:0;font-weight:400;}}
.xray-wrap{{
    background:{xray_wrap_bg};border:1px solid {bdr};
    border-radius:14px;padding:12px;overflow:hidden;
}}
.info-hint{{
    background:{badge_bg};border:1px solid {badge_bdr};
    border-radius:8px;padding:10px 14px;
    font-size:0.8rem;color:{muted_text};font-weight:500;
}}

/* ── Validity breakdown panel ─────────────────── */
.validity-panel{{
    background:{bg_card};border:1px solid {card_bdr};
    border-radius:12px;padding:1.2rem 1.4rem;margin-top:1rem;
}}
.validity-panel h5{{
    font-size:0.65rem;text-transform:uppercase;letter-spacing:2.5px;
    color:{muted_text};font-weight:700;margin:0 0 0.8rem 0;
}}
.validity-row{{
    display:flex;justify-content:space-between;align-items:center;
    font-size:0.78rem;padding:5px 0;border-bottom:1px solid {bdr};color:{body_text};
}}
.validity-row:last-child{{border-bottom:none;}}
.validity-pass{{color:#22aa55;font-weight:700;font-size:0.75rem;}}
.validity-fail{{color:#cc2222;font-weight:700;font-size:0.75rem;}}
.validity-warn{{color:#cc9900;font-weight:700;font-size:0.75rem;}}

.stTextInput>div>div>input{{
    background:{input_bg} !important;border:1px solid {input_bdr} !important;
    color:{input_color} !important;border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important;font-size:0.88rem !important;
    font-weight:400 !important;padding:0.55rem 0.9rem !important;
}}
.stTextInput label{{
    color:{body_text} !important;font-size:0.78rem !important;
    letter-spacing:0.5px;font-weight:600 !important;text-transform:uppercase;
}}
.stButton>button{{
    background:linear-gradient(135deg,{'#002d44,#004466' if dark else '#0066aa,#0088cc'}) !important;
    color:{'#00c8f0' if dark else '#ffffff'} !important;
    border:1px solid {'#00c8f033' if dark else '#0077bb'} !important;
    border-radius:10px !important;font-family:'DM Sans',sans-serif !important;
    font-size:0.84rem !important;font-weight:700 !important;
    padding:0.6rem 1.4rem !important;width:100% !important;
    letter-spacing:0.3px !important;transition:all 0.2s ease !important;
}}
.stDownloadButton>button{{
    background:linear-gradient(135deg,#0d4a22,#157a32) !important;
    color:#ffffff !important;border:1px solid #2a9944 !important;
    border-radius:10px !important;font-family:'DM Sans',sans-serif !important;
    font-size:0.84rem !important;font-weight:700 !important;width:100% !important;
    padding:0.6rem 1.4rem !important;letter-spacing:0.3px !important;
}}
div[data-testid="stFileUploader"]{{
    background:{input_bg};border:2px dashed {bdr};border-radius:14px;padding:0.8rem;
}}
hr{{border-color:{hr_color} !important;border-width:1px 0 0 0 !important;}}
[data-testid="stMarkdownContainer"] p{{color:{body_text};font-weight:400;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-thumb{{background:{bdr};border-radius:4px;}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_puma_model():
    if not os.path.exists("puma_cnn_model.h5"):
        return None, None
    model  = load_model("puma_cnn_model.h5")
    bundle = None
    if os.path.exists("puma_model_bundle.pkl"):
        with open("puma_model_bundle.pkl","rb") as f:
            bundle = pickle.load(f)
    return model, bundle


# ══════════════════════════════════════════════════════
# IMPROVED X-RAY VALIDITY CHECKER
# ══════════════════════════════════════════════════════
def check_xray_validity(img_pil):
    """
    Multi-criteria validity checker designed to reject non-X-ray images
    before they reach the CNN.

    Criteria assessed:
      1. Grayscale dominance  — real X-rays are monochrome; colour photos fail
      2. Mean brightness      — X-rays are mid-tone (40–200); too bright = photo
      3. Tonal spread (std)   — X-rays have wide tonal range; flat images fail
      4. Aspect ratio         — chest X-rays are roughly square to portrait
      5. Dark-pixel ratio     — X-rays have a meaningful proportion of dark pixels
      6. High-saturation ratio— vivid colours signal non-X-ray images
      7. Edge density         — X-rays have clear structural edges (ribs, lungs)

    Returns:
      is_valid (bool), score (int 0-100), reason (str|None), details (dict)
    """
    img_rgb  = np.array(img_pil.convert("RGB"),  dtype=np.float32)
    img_gray = np.array(img_pil.convert("L"),    dtype=np.float32)

    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    details = {}
    deductions = []

    # ── 1. Grayscale dominance ────────────────────────────────────
    # Mean absolute channel differences; real X-rays score < 8
    rg_diff = float(np.mean(np.abs(R - G)))
    gb_diff = float(np.mean(np.abs(G - B)))
    rb_diff = float(np.mean(np.abs(R - B)))
    color_diff = (rg_diff + gb_diff + rb_diff) / 3.0
    details["color_diff"] = round(color_diff, 2)

    if color_diff > 30:
        deductions.append(("Colour photograph detected", 60))
        details["color_diff_status"] = "FAIL"
    elif color_diff > 15:
        deductions.append(("Slight colour tint present", 20))
        details["color_diff_status"] = "WARN"
    else:
        details["color_diff_status"] = "PASS"

    # ── 2. Mean brightness ────────────────────────────────────────
    # Real chest X-rays: mean ~80–185; photos are often brighter
    mean_bright = float(np.mean(img_gray))
    details["mean_brightness"] = round(mean_bright, 1)

    if mean_bright > 210:
        deductions.append(("Image too bright (overexposed or colour photo)", 35))
        details["brightness_status"] = "FAIL"
    elif mean_bright < 15:
        deductions.append(("Image too dark (near-black image)", 30))
        details["brightness_status"] = "FAIL"
    elif mean_bright > 195 or mean_bright < 25:
        deductions.append(("Unusual brightness for a chest X-ray", 15))
        details["brightness_status"] = "WARN"
    else:
        details["brightness_status"] = "PASS"

    # ── 3. Tonal spread (standard deviation) ─────────────────────
    # X-rays have rich tonal variation (std > 35 typically)
    std_bright = float(np.std(img_gray))
    details["tonal_spread"] = round(std_bright, 1)

    if std_bright < 20:
        deductions.append(("Flat image — insufficient tonal range for X-ray", 40))
        details["tonal_status"] = "FAIL"
    elif std_bright < 30:
        deductions.append(("Low tonal variance — may not be an X-ray", 15))
        details["tonal_status"] = "WARN"
    else:
        details["tonal_status"] = "PASS"

    # ── 4. Aspect ratio ───────────────────────────────────────────
    # PA chest X-ray: roughly 0.7 – 1.5 (width/height)
    w, h = img_pil.size
    ratio = w / h
    details["aspect_ratio"] = round(ratio, 2)

    if ratio > 2.8 or ratio < 0.25:
        deductions.append(("Extreme aspect ratio — unlikely chest X-ray", 30))
        details["ratio_status"] = "FAIL"
    elif ratio > 1.8 or ratio < 0.45:
        deductions.append(("Unusual aspect ratio for a chest X-ray", 12))
        details["ratio_status"] = "WARN"
    else:
        details["ratio_status"] = "PASS"

    # ── 5. Dark-pixel ratio ───────────────────────────────────────
    # Chest X-rays typically have 20-65 % dark pixels (< 80 intensity)
    dark_ratio = float(np.mean(img_gray < 80))
    details["dark_pixel_ratio"] = round(dark_ratio, 3)

    if dark_ratio < 0.05:
        deductions.append(("Too few dark pixels — image appears washed out", 25))
        details["dark_ratio_status"] = "FAIL"
    elif dark_ratio > 0.85:
        deductions.append(("Too many dark pixels — near-black image", 25))
        details["dark_ratio_status"] = "FAIL"
    else:
        details["dark_ratio_status"] = "PASS"

    # ── 6. High-saturation pixel ratio ───────────────────────────
    # Convert to HSV and check proportion of vivid pixels
    # X-rays should have almost zero highly saturated pixels
    img_hsv = np.array(img_pil.convert("RGB").convert("RGB"))
    # Manual saturation approximation: (max - min) / max per pixel
    pmax = np.max(img_rgb, axis=2)
    pmin = np.min(img_rgb, axis=2)
    sat  = np.where(pmax > 0, (pmax - pmin) / (pmax + 1e-6), 0.0)
    high_sat_ratio = float(np.mean(sat > 0.35))
    details["high_saturation_ratio"] = round(high_sat_ratio, 4)

    if high_sat_ratio > 0.15:
        deductions.append(("High colour saturation detected — not an X-ray", 55))
        details["saturation_status"] = "FAIL"
    elif high_sat_ratio > 0.06:
        deductions.append(("Moderate colour saturation — questionable X-ray", 20))
        details["saturation_status"] = "WARN"
    else:
        details["saturation_status"] = "PASS"

    # ── 7. Edge density ───────────────────────────────────────────
    # X-rays contain clear structural edges (ribs, diaphragm, lungs)
    # We use a simple Sobel-style gradient on a downsampled version
    thumb = np.array(img_pil.convert("L").resize((128, 128)), dtype=np.float32)
    gx = np.abs(np.diff(thumb, axis=1))
    gy = np.abs(np.diff(thumb, axis=0))
    edge_density = float(np.mean(gx) + np.mean(gy)) / 2.0
    details["edge_density"] = round(edge_density, 2)

    if edge_density < 3.0:
        deductions.append(("Very low edge density — insufficient structural detail", 30))
        details["edge_status"] = "FAIL"
    elif edge_density < 5.0:
        deductions.append(("Low edge density — structural features not clear", 10))
        details["edge_status"] = "WARN"
    else:
        details["edge_status"] = "PASS"

    # ── Compute final score ───────────────────────────────────────
    score = 100
    primary_reason = None
    for reason_text, penalty in deductions:
        score -= penalty
        if primary_reason is None:
            primary_reason = reason_text

    score = max(0, score)

    # Threshold: must score ≥ 55 to proceed (raised from original 40)
    is_valid = (score >= 55)

    return is_valid, score, primary_reason, details


# ══════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════
def preprocess_image(img_pil, img_size=224):
    img = img_pil.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ══════════════════════════════════════════════════════
# PREDICTION  (with confidence gating)
# ══════════════════════════════════════════════════════
def predict(model, img_array):
    """
    Run inference. If the model's raw confidence is below a minimum
    threshold we flag the result as uncertain rather than blindly
    outputting the majority-class label.
    """
    prob_p = float(model.predict(img_array, verbose=0)[0][0])
    prob_n = 1.0 - prob_p

    # Confidence gating: require ≥ 60 % confidence for a firm label
    CONFIDENCE_GATE = 0.60

    if prob_p >= CONFIDENCE_GATE:
        label = "PNEUMONIA"
        conf  = prob_p
    elif prob_n >= CONFIDENCE_GATE:
        label = "NORMAL"
        conf  = prob_n
    else:
        # Both classes below gate → uncertain
        label = "UNCERTAIN"
        conf  = max(prob_p, prob_n)

    return label, conf, {"NORMAL": prob_n, "PNEUMONIA": prob_p}


# ══════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════
def get_recommendations(label, conf):
    if label == "PNEUMONIA" and conf >= 0.80:
        return "pneumonia", "High-Confidence Pneumonia Detected", [
            "Seek medical attention promptly — present this report to a qualified physician or pulmonologist.",
            "Do not self-medicate. A doctor will determine whether the cause is bacterial or viral.",
            "Monitor symptoms: worsening breathlessness, high fever, or chest pain warrants emergency care.",
            "Rest, hydrate, and avoid strenuous activity until cleared by a medical professional.",
            "A follow-up chest X-ray may be required after treatment to confirm resolution.",
        ]
    elif label == "PNEUMONIA" and conf >= 0.60:
        return "pneumonia", "Possible Pneumonia — Consult a Doctor", [
            "Model confidence is moderate. This result should be reviewed by a licensed physician.",
            "Describe your symptoms clearly: duration of cough, fever pattern, and any breathing difficulty.",
            "Your doctor may order additional tests (blood work, CT scan) for a definitive diagnosis.",
            "Avoid crowded spaces and take standard precautions while awaiting professional evaluation.",
        ]
    elif label == "NORMAL" and conf >= 0.80:
        return "normal", "No Pneumonia Detected", [
            "The X-ray appears normal. No significant pulmonary infiltrates were identified by the model.",
            "If you still have respiratory symptoms, consult a physician — not all conditions show on X-ray.",
            "Maintain good respiratory hygiene: handwashing, avoiding smoke, and keeping vaccinations current.",
            "Schedule routine health checkups as advised by your healthcare provider.",
        ]
    else:
        return "uncertain", "Low Confidence — Inconclusive Result", [
            "The model could not make a confident determination from this image.",
            "Image quality or positioning may be affecting the result. Retake the X-ray if possible.",
            "This result must be reviewed by a qualified radiologist before any clinical decision is made.",
            "Do not rely on this output alone. Seek professional medical evaluation.",
        ]


# ══════════════════════════════════════════════════════
# PDF BUILDER
# ══════════════════════════════════════════════════════
def hex_to_rl(hx):
    hx = hx.lstrip('#')
    return colors.Color(*[int(hx[i:i+2], 16) / 255 for i in (0, 2, 4)])


def draw_header(c, W, H, HDR_H, white, navy, accent, slate_lt):
    c.setFillColor(navy); c.rect(0, H-HDR_H, W, HDR_H, fill=1, stroke=0)
    c.setFillColor(accent); c.rect(W-5*mm, H-HDR_H, 5*mm, HDR_H, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 13); c.setFillColor(white)
    c.drawString(18*mm, H-12*mm, "Bells University of Technology")
    c.setFont("Helvetica", 7.5); c.setFillColor(hex_to_rl("9ab8d8"))
    c.drawString(18*mm, H-19*mm,
        "Department of Computer Science & Information Technology  |  College of Natural & Applied Sciences, Ota")
    c.setStrokeColor(colors.Color(1, 1, 1, 0.12)); c.setLineWidth(0.4)
    c.line(18*mm, H-23*mm, W-24*mm, H-23*mm)
    c.setFont("Helvetica-Bold", 9); c.setFillColor(white)
    c.drawString(18*mm, H-30*mm, "PneumoScan  ·  PUMA-Optimized CNN")
    c.setFont("Helvetica", 7); c.setFillColor(slate_lt)
    c.drawRightString(W-26*mm, H-30*mm, "AI-ASSISTED CHEST X-RAY RADIOLOGY REPORT")


def draw_footer(c, W, page_num, total, date_str, time_str, navy, slate):
    FY = 16*mm
    c.setStrokeColor(navy); c.setLineWidth(0.6)
    c.line(18*mm, FY, W-18*mm, FY)
    c.setFont("Helvetica-Bold", 7.5); c.setFillColor(navy)
    c.drawString(18*mm, FY-5*mm, "PneumoScan  |  AI Radiology Screening System")
    c.setFont("Helvetica", 6.5); c.setFillColor(slate)
    c.drawString(18*mm, FY-10*mm,
        "Bells University of Technology, Ota  |  B.Tech Computer Science — Final Year Project")
    c.setFont("Helvetica", 7); c.setFillColor(slate)
    c.drawRightString(W-18*mm, FY-5*mm,  f"{date_str}  ·  {time_str}")
    c.drawRightString(W-18*mm, FY-10*mm, f"Page {page_num} of {total}")
    c.setFillColor(navy)
    c.rect(0, 0, 5*mm, FY+2*mm, fill=1, stroke=0)
    c.rect(W-5*mm, 0, 5*mm, FY+2*mm, fill=1, stroke=0)


def sec_head(c, x, y, text, navy):
    c.setFont("Helvetica-Bold", 8.5); c.setFillColor(navy)
    c.drawString(x, y, text)
    y -= 2*mm; c.setStrokeColor(navy); c.setLineWidth(1.0)
    c.line(x, y, x+65*mm, y)
    return y - 6*mm


def wrap_text(c, text, max_w, font="Helvetica", size=8.5):
    words = text.split(); line = ""; lines = []
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, size) < max_w:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return lines


def build_pdf(first, last, email, label, conf, probs, img_pil, bundle, rec_items):
    buf = io.BytesIO()
    W, H = A4
    c = rl_canvas.Canvas(buf, pagesize=A4)

    white   = colors.white
    pg_bg   = hex_to_rl("fafcff")
    navy    = hex_to_rl("0b1f3d")
    navy_lt = hex_to_rl("1c3d7a")
    slate   = hex_to_rl("4a6282")
    slt_lt  = hex_to_rl("8a9db8")
    rule    = hex_to_rl("c8d8ea")
    shade   = hex_to_rl("f2f6fa")
    black   = hex_to_rl("1a1e2a")
    red_c   = hex_to_rl("b31c1c"); grn_c = hex_to_rl("1a7a3c"); amb_c = hex_to_rl("8a6400")
    red_bg  = hex_to_rl("fff2f2"); grn_bg = hex_to_rl("f2fbf5"); amb_bg = hex_to_rl("fffbf0")
    red_bdr = hex_to_rl("e8a0a0"); grn_bdr = hex_to_rl("80c89a"); amb_bdr = hex_to_rl("d4b464")
    accent  = hex_to_rl("0088cc")
    warn_bg = hex_to_rl("fff8e5"); warn_bdr = hex_to_rl("c89a18")

    rc = red_c   if label == "PNEUMONIA" else (grn_c   if label == "NORMAL" else amb_c)
    rb = red_bg  if label == "PNEUMONIA" else (grn_bg  if label == "NORMAL" else amb_bg)
    rd = red_bdr if label == "PNEUMONIA" else (grn_bdr if label == "NORMAL" else amb_bdr)

    now      = datetime.datetime.now()
    date_str = now.strftime("%d %B %Y")
    time_str = now.strftime("%H:%M UTC")
    rep_id   = f"PSR-{now.strftime('%Y%m%d%H%M%S')}"

    verdict = (
        "PNEUMONIA DETECTED"
        if label == "PNEUMONIA" else
        "NO PNEUMONIA DETECTED"
        if label == "NORMAL" else
        "INCONCLUSIVE — FURTHER REVIEW REQUIRED"
    )
    impression = (
        "Radiographic findings are consistent with pneumonia. Pulmonary infiltrates detected with high model confidence. Clinical correlation and physician review are strongly advised."
        if label == "PNEUMONIA" and conf >= 0.80 else
        "Radiographic findings suggest possible pneumonia. Model confidence is moderate; physician review and supplementary investigations recommended."
        if label == "PNEUMONIA" else
        "No radiographic evidence of pneumonia identified. Lung fields appear clear. Clinical correlation recommended if symptoms persist."
        if label == "NORMAL" and conf >= 0.80 else
        "Analysis is inconclusive due to low model confidence. Image quality or positioning may be suboptimal. Radiologist review is required."
    )

    HDR_H = 38*mm

    # ── PAGE 1 ────────────────────────────────────────────────────
    c.setFillColor(pg_bg); c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_header(c, W, H, HDR_H, white, navy, accent, slt_lt)

    y = H - HDR_H - 5*mm
    c.setFont("Helvetica-Bold", 11.5); c.setFillColor(navy)
    c.drawString(18*mm, y, "RADIOLOGY REPORT — CHEST X-RAY ANALYSIS")
    c.setFont("Helvetica", 7.5); c.setFillColor(slate)
    c.drawRightString(W-18*mm, y, f"Report ID: {rep_id}")
    y -= 3*mm; c.setStrokeColor(rule); c.setLineWidth(0.8)
    c.line(18*mm, y, W-18*mm, y); y -= 8*mm

    INFO_H = 32*mm
    c.setFillColor(shade); c.roundRect(18*mm, y-INFO_H, W-36*mm, INFO_H, 3, fill=1, stroke=0)
    c.setStrokeColor(rule); c.setLineWidth(0.5)
    c.roundRect(18*mm, y-INFO_H, W-36*mm, INFO_H, 3, fill=0, stroke=1)
    c.setFillColor(navy_lt); c.rect(18*mm, y-INFO_H, 3*mm, INFO_H, fill=1, stroke=0)

    col_l = 18*mm + 3*mm + 6*mm
    col_r = 18*mm + (W-36*mm)/2 + 8*mm
    y_i   = y - 7*mm
    rows  = [
        ("PATIENT NAME",   f"{first} {last}",    "REPORT DATE",       date_str),
        ("EMAIL ADDRESS",  email,                 "TIME OF ANALYSIS",  time_str),
        ("REPORT ID",      rep_id,                "IMAGING MODALITY",  "Chest X-Ray (PA/AP)"),
    ]
    for lbl_l, val_l, lbl_r, val_r in rows:
        c.setFont("Helvetica", 6.5); c.setFillColor(slt_lt)
        c.drawString(col_l, y_i, lbl_l); c.drawString(col_r, y_i, lbl_r)
        c.setFont("Helvetica-Bold", 9); c.setFillColor(black)
        c.drawString(col_l, y_i-5*mm, val_l); c.drawString(col_r, y_i-5*mm, val_r)
        y_i -= 9.5*mm

    y -= INFO_H + 9*mm
    y = sec_head(c, 18*mm, y, "CLINICAL IMPRESSION", navy)

    VH = 20*mm
    c.setFillColor(rb); c.roundRect(18*mm, y-VH, W-36*mm, VH, 4, fill=1, stroke=0)
    c.setStrokeColor(rd); c.setLineWidth(0.6)
    c.roundRect(18*mm, y-VH, W-36*mm, VH, 4, fill=0, stroke=1)
    c.setFillColor(rc); c.rect(18*mm, y-VH, 3.5*mm, VH, fill=1, stroke=0)

    BW = 22*mm; BH = 10*mm
    bx = W - 18*mm - BW - 3*mm; by = y - VH/2 - BH/2
    c.setFillColor(rc); c.roundRect(bx, by, BW, BH, 3, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10); c.setFillColor(white)
    c.drawCentredString(bx+BW/2, by+3*mm, f"{conf*100:.1f}%")

    tx = 18*mm + 3.5*mm + 5*mm
    c.setFont("Helvetica-Bold", 11); c.setFillColor(rc); c.drawString(tx, y-7*mm, verdict)
    c.setFont("Helvetica", 7.5); c.setFillColor(slate)
    c.drawString(tx, y-13.5*mm, f"Confidence: {conf*100:.1f}%   |   PUMA-Optimized CNN v1.0   |   {date_str}")

    y -= VH + 6*mm
    c.setFont("Helvetica-Bold", 7.5); c.setFillColor(slate); c.drawString(18*mm, y, "Impression:")
    y -= 5*mm; c.setFont("Helvetica", 8.5); c.setFillColor(black)
    for ln in wrap_text(c, impression, W-36*mm):
        c.drawString(18*mm, y, ln); y -= 5.5*mm
    y -= 8*mm

    y = sec_head(c, 18*mm, y, "PROBABILITY ANALYSIS", navy)
    LW = 52*mm; PW = 20*mm; BW2 = W - 36*mm - LW - PW - 8*mm; bx2 = 18*mm + LW
    for lbl_t, pct, col, bg in [
        ("Pneumonia Probability", probs["PNEUMONIA"], red_c, red_bg),
        ("Normal Probability",    probs["NORMAL"],    grn_c, grn_bg),
    ]:
        c.setFont("Helvetica", 8.5); c.setFillColor(black); c.drawString(18*mm, y, lbl_t)
        c.setFillColor(rule); c.roundRect(bx2, y-3.5*mm, BW2, 5*mm, 2, fill=1, stroke=0)
        if pct > 0.005:
            c.setFillColor(col); c.roundRect(bx2, y-3.5*mm, max(3, BW2*pct), 5*mm, 2, fill=1, stroke=0)
        px = bx2 + BW2 + 4*mm
        c.setFillColor(bg); c.roundRect(px, y-3*mm, PW, 4.5*mm, 2, fill=1, stroke=0)
        c.setStrokeColor(col); c.setLineWidth(0.4)
        c.roundRect(px, y-3*mm, PW, 4.5*mm, 2, fill=0, stroke=1)
        c.setFont("Helvetica-Bold", 8); c.setFillColor(col)
        c.drawCentredString(px+PW/2, y-0.5*mm, f"{pct*100:.1f}%")
        y -= 13*mm
    y -= 6*mm

    y = sec_head(c, 18*mm, y, "RADIOGRAPHIC IMAGE", navy)
    ICW = (W-40*mm)*0.52; NW = (W-40*mm)*0.48 - 4*mm; IH = 65*mm; nx = 18*mm + ICW + 4*mm

    c.setFillColor(hex_to_rl("0a0e14")); c.roundRect(18*mm, y-IH, ICW, IH, 4, fill=1, stroke=0)
    c.setStrokeColor(rule); c.setLineWidth(0.5); c.roundRect(18*mm, y-IH, ICW, IH, 4, fill=0, stroke=1)
    ob = io.BytesIO(); img_pil.convert("RGB").save(ob, format="JPEG", quality=92); ob.seek(0)
    c.drawImage(ImageReader(ob), 18*mm+2*mm, y-IH+2*mm, width=ICW-4*mm, height=IH-4*mm, preserveAspectRatio=True)
    c.setFont("Helvetica", 6.5); c.setFillColor(slate)
    c.drawCentredString(18*mm+ICW/2, y-IH-4*mm, "Figure 1 — Submitted Chest X-Ray")

    c.setFillColor(shade); c.roundRect(nx, y-IH, NW, IH, 4, fill=1, stroke=0)
    c.setStrokeColor(rule); c.setLineWidth(0.5); c.roundRect(nx, y-IH, NW, IH, 4, fill=0, stroke=1)
    c.setFillColor(navy_lt); c.rect(nx, y-IH, 2.5*mm, IH, fill=1, stroke=0)
    ny = y - 7*mm
    c.setFont("Helvetica-Bold", 7); c.setFillColor(navy); c.drawString(nx+5*mm, ny, "AI ANALYSIS NOTES")
    ny -= 1.5*mm; c.setStrokeColor(slt_lt); c.setLineWidth(0.3)
    c.line(nx+5*mm, ny, nx+NW-3*mm, ny); ny -= 5.5*mm

    anotes = [
        ("Analysis Engine",   "PUMA-CNN v1.0"),
        ("Architecture",      "Convolutional Neural Net"),
        ("Optimizer",         "PUMA Metaheuristic"),
        ("Input Resolution",  "224 × 224 px (RGB)"),
        ("Output Layer",      "Sigmoid (Binary)"),
        ("Prediction",        label.title()),
        ("Confidence",        f"{conf*100:.1f}%"),
    ]
    if bundle:
        m = bundle.get("metrics", {})
        anotes += [
            ("Model Accuracy", f"{m.get('accuracy',  0)*100:.1f}%"),
            ("Precision",      f"{m.get('precision', 0)*100:.1f}%"),
            ("Recall",         f"{m.get('recall',    0)*100:.1f}%"),
            ("F1 Score",       f"{m.get('f1_score',  0)*100:.1f}%"),
        ]
    for kt, vt in anotes:
        if ny < y - IH + 5*mm: break
        c.setFont("Helvetica", 6.5); c.setFillColor(slt_lt); c.drawString(nx+5*mm, ny, kt)
        c.setFont("Helvetica-Bold", 6.5); c.setFillColor(black); c.drawRightString(nx+NW-4*mm, ny, vt)
        ny -= 4*mm; c.setStrokeColor(rule); c.setLineWidth(0.2)
        c.line(nx+5*mm, ny+0.5*mm, nx+NW-4*mm, ny+0.5*mm); ny -= 2*mm

    draw_footer(c, W, 1, 2, date_str, time_str, navy, slate)
    c.showPage()

    # ── PAGE 2 ────────────────────────────────────────────────────
    c.setFillColor(pg_bg); c.rect(0, 0, W, H, fill=1, stroke=0)
    draw_header(c, W, H, HDR_H, white, navy, accent, slt_lt)

    y = H - HDR_H - 5*mm
    c.setFont("Helvetica-Bold", 10); c.setFillColor(navy)
    c.drawString(18*mm, y, "RADIOLOGY REPORT — CONTINUED")
    c.setFont("Helvetica", 7.5); c.setFillColor(slate)
    c.drawRightString(W-18*mm, y, f"Report ID: {rep_id}")
    y -= 3*mm; c.setStrokeColor(rule); c.setLineWidth(0.8)
    c.line(18*mm, y, W-18*mm, y); y -= 12*mm

    y = sec_head(c, 18*mm, y, "CLINICAL RECOMMENDATIONS", navy)
    REC_ITEM_BASE = 14*mm
    total_rec_h   = 8*mm
    for item in rec_items:
        ls = wrap_text(c, item, W-36*mm-18*mm, "Helvetica", 8.5)
        total_rec_h += max(REC_ITEM_BASE, len(ls)*5.5*mm + 6*mm)

    c.setFillColor(rb); c.roundRect(18*mm, y-total_rec_h, W-36*mm, total_rec_h, 4, fill=1, stroke=0)
    c.setStrokeColor(rd); c.setLineWidth(0.5)
    c.roundRect(18*mm, y-total_rec_h, W-36*mm, total_rec_h, 4, fill=0, stroke=1)
    c.setFillColor(rc); c.rect(18*mm, y-total_rec_h, 3.5*mm, total_rec_h, fill=1, stroke=0)

    ry = y - 7*mm
    for i, item in enumerate(rec_items, 1):
        ls = wrap_text(c, item, W-36*mm-18*mm, "Helvetica", 8.5)
        c.setFont("Helvetica-Bold", 9); c.setFillColor(rc)
        c.drawString(24*mm, ry, f"{i}.")
        c.setFont("Helvetica", 8.5); c.setFillColor(black)
        for li, ln in enumerate(ls):
            c.drawString(31*mm, ry-li*5.5*mm, ln)
        ry -= max(REC_ITEM_BASE, len(ls)*5.5*mm + 6*mm)

    y -= total_rec_h + 14*mm
    y = sec_head(c, 18*mm, y, "NEXT STEPS", navy)
    steps = [
        ("🏥", "Visit a Healthcare Provider",
         "Take this report to your nearest hospital, clinic, or physician for professional evaluation and confirmation."),
        ("🔬", "Await Clinical Diagnosis",
         "This AI result is a screening aid only. Your doctor will confirm through physical examination and additional tests."),
        ("📋", "Keep Your Records",
         "Store this report with your medical documents. The Report ID supports traceability and follow-up discussions."),
    ]
    SW = (W-36*mm-8*mm)/3; sx = 18*mm; SH = 32*mm
    for icon, title, desc in steps:
        c.setFillColor(shade); c.roundRect(sx, y-SH, SW, SH, 4, fill=1, stroke=0)
        c.setStrokeColor(rule); c.setLineWidth(0.4)
        c.roundRect(sx, y-SH, SW, SH, 4, fill=0, stroke=1)
        c.setFillColor(navy_lt); c.rect(sx, y-SH, SW, 2.5*mm, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 14); c.setFillColor(navy); c.drawString(sx+5*mm, y-9*mm, icon)
        c.setFont("Helvetica-Bold", 7.5); c.setFillColor(navy); c.drawString(sx+5*mm, y-15*mm, title)
        dls = wrap_text(c, desc, SW-10*mm, "Helvetica", 7)
        c.setFont("Helvetica", 7); c.setFillColor(slate)
        for di, dl in enumerate(dls):
            c.drawString(sx+5*mm, y-21*mm-di*4.5*mm, dl)
        sx += SW + 4*mm

    y -= SH + 14*mm
    DH = 24*mm
    c.setFillColor(warn_bg); c.roundRect(18*mm, y-DH, W-36*mm, DH, 4, fill=1, stroke=0)
    c.setStrokeColor(warn_bdr); c.setLineWidth(0.5)
    c.roundRect(18*mm, y-DH, W-36*mm, DH, 4, fill=0, stroke=1)
    c.setFillColor(warn_bdr); c.rect(18*mm, y-DH, 3.5*mm, DH, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 7.5); c.setFillColor(hex_to_rl("8a6200"))
    c.drawString(25*mm, y-6*mm, "IMPORTANT DISCLAIMER")
    c.setFont("Helvetica", 7.5); c.setFillColor(hex_to_rl("5a4400"))
    disc = [
        "This report is generated by an AI-assisted screening tool for academic and research purposes only.",
        "It does NOT constitute a medical diagnosis or replace the advice of a qualified physician or radiologist.",
        "All findings must be reviewed by a licensed clinician before any clinical decision is made.",
        "Bells University of Technology assumes no liability for clinical decisions based on this report.",
    ]
    for di, dl in enumerate(disc):
        c.drawString(25*mm, y-12*mm-di*4.5*mm, dl)

    draw_footer(c, W, 2, 2, date_str, time_str, navy, slate)
    c.save(); buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class='brand-wrap'>
        <div class='brand-icon'>🫁</div>
        <div class='brand-name'>PneumoScan</div>
        <div class='brand-sub'>PUMA-CNN System</div>
    </div>
    <hr style='border-color:{hr_color};margin:0 0 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Analyze", "PUMA Insights", "About"], label_visibility="hidden")

    st.markdown(f"<hr style='border-color:{hr_color};margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.button(f"{toggle_icon}  {toggle_label}", on_click=toggle_theme)

    st.markdown(f"""
    <div style='margin-top:1.2rem;padding:1rem;background:{sidebar_pill};
         border:1px solid {sidebar_pill_bdr};border-radius:10px;'>
        <div style='font-size:0.58rem;letter-spacing:2px;text-transform:uppercase;
             color:{muted_text};font-weight:700;margin-bottom:8px;'>Researcher</div>
        <div style='font-size:0.83rem;color:{sidebar_text};font-weight:600;line-height:1.9;'>
            Olatunji Olusegun Peters<br>
            <span style='font-size:0.75rem;font-weight:400;color:{muted_text};'>Matric: 2021/10103</span>
        </div>
        <div style='font-size:0.72rem;color:{muted_text};line-height:1.8;margin-top:6px;'>
            B.Tech Computer Science<br>Bells University of Technology
        </div>
    </div>
    """, unsafe_allow_html=True)


model, bundle = load_puma_model()


# ══════════════════════════════════════════════════════
# PAGE: Analyze
# ══════════════════════════════════════════════════════
if page == "Analyze":
    st.markdown(f"""
    <div class='page-hero'>
        <div class='hero-badge'><div class='hero-dot'></div>PUMA-CNN Active</div>
        <h1>Chest X-Ray<br><em>Analysis</em></h1>
        <p>Upload a chest X-ray for PUMA-CNN pneumonia screening with confidence scores and clinical recommendations.</p>
    </div>
    <div class='hero-rule'></div><br>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div class='warning-banner'>
            <h4>⚠ Model Not Found</h4>
            <p>Place <strong>puma_cnn_model.h5</strong> and <strong>puma_model_bundle.pkl</strong>
            in the same directory as app.py, then restart.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("<div class='section-label'>Patient Details</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1: first = st.text_input("First Name",     placeholder="e.g. Olusegun")
    with c2: last  = st.text_input("Last Name",      placeholder="e.g. Peters")
    with c3: email = st.text_input("Email Address",  placeholder="e.g. patient@hospital.com")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Upload X-Ray Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded:
        img_pil = Image.open(uploaded)

        # ── Run improved validity check ───────────────────────────
        is_valid, validity_score, invalid_reason, validity_details = check_xray_validity(img_pil)

        if not is_valid:
            # ── Rejection UI ──────────────────────────────────────
            st.markdown(f"""
            <div class='warning-banner'>
                <h4>⚠ Image Rejected — Not a Valid Chest X-Ray</h4>
                <p><strong>Reason:</strong> {invalid_reason}<br>
                Please upload a proper grayscale frontal chest X-ray (PA or AP view).
                Colour photographs, screenshots, or unrelated medical images cannot be analysed.</p>
            </div>
            """, unsafe_allow_html=True)

            cp, cm = st.columns([1, 2])
            with cp:
                st.markdown("<div class='xray-wrap'>", unsafe_allow_html=True)
                st.image(img_pil, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with cm:
                st.markdown(f"""
                <div style='padding:1.8rem;'>
                    <div class='section-label'>Validity Check</div>
                    <div style='font-family:"Fraunces",serif;font-size:1.9rem;color:#aa7700;
                         margin:0.6rem 0;font-weight:700;letter-spacing:-1px;'>Not an X-Ray</div>
                    <div style='display:inline-flex;align-items:center;gap:8px;
                         background:{unc_bg};border:1px solid {unc_bdr};border-radius:8px;
                         padding:6px 14px;font-size:0.82rem;font-weight:700;color:#aa7700;'>
                        Validity Score: {validity_score}/100
                    </div>
                    <div style='font-size:0.85rem;color:{body_text};margin-top:1.2rem;line-height:1.75;'>
                        Please upload a frontal chest radiograph for accurate analysis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Show per-criterion breakdown ──────────────────
                def status_badge(s):
                    if s == "PASS":
                        return "<span class='validity-pass'>✔ PASS</span>"
                    elif s == "FAIL":
                        return "<span class='validity-fail'>✘ FAIL</span>"
                    return "<span class='validity-warn'>⚠ WARN</span>"

                rows_html = "".join([
                    f"<div class='validity-row'><span>{label_}</span>{status_badge(val_)}</div>"
                    for label_, key_ in [
                        ("Grayscale dominance",    "color_diff_status"),
                        ("Brightness range",        "brightness_status"),
                        ("Tonal spread",            "tonal_status"),
                        ("Aspect ratio",            "ratio_status"),
                        ("Dark-pixel ratio",        "dark_ratio_status"),
                        ("Colour saturation",       "saturation_status"),
                        ("Edge / structural detail","edge_status"),
                    ]
                    for val_ in [validity_details.get(key_, "PASS")]
                ])
                st.markdown(f"""
                <div class='validity-panel'>
                    <h5>Validity Criteria Breakdown</h5>
                    {rows_html}
                </div>
                """, unsafe_allow_html=True)

        else:
            # ── Valid image: run inference ─────────────────────────
            with st.spinner("Analysing with PUMA-CNN…"):
                img_size = bundle["img_size"] if bundle else 224
                img_arr  = preprocess_image(img_pil, img_size)
                label, conf, probs = predict(model, img_arr)
                rec_type, rec_title, rec_items = get_recommendations(label, conf)

            st.markdown("<br>", unsafe_allow_html=True)
            ci, cr = st.columns([1, 1.15])

            with ci:
                st.markdown("<div class='section-label'>Original X-Ray</div>", unsafe_allow_html=True)
                st.markdown("<div class='xray-wrap'>", unsafe_allow_html=True)
                st.image(img_pil, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Show validity score for accepted images too
                st.markdown(f"""
                <div style='margin-top:0.8rem;padding:8px 14px;background:{badge_bg};
                     border:1px solid {badge_bdr};border-radius:8px;font-size:0.75rem;
                     color:{hero_accent};font-weight:600;text-align:center;'>
                    ✔ X-Ray Validity Score: {validity_score}/100
                </div>
                """, unsafe_allow_html=True)

            with cr:
                st.markdown("<div class='section-label'>Diagnosis Result</div>", unsafe_allow_html=True)

                icon    = "🔴" if label == "PNEUMONIA" else ("🟢" if label == "NORMAL" else "🟡")
                verdict = (
                    "Pneumonia Detected" if label == "PNEUMONIA"
                    else "Normal"         if label == "NORMAL"
                    else "Uncertain"
                )

                st.markdown(f"""
                <div class='result-card {rec_type}'>
                    <span class='icon'>{icon}</span>
                    <div class='sublabel'>Prediction</div>
                    <div class='verdict'>{verdict}</div>
                    <div class='divider'></div>
                    <div class='score'>{conf*100:.1f}<span style='font-size:1.4rem;'>%</span></div>
                    <div class='sublabel'>Model Confidence</div>
                </div>
                """, unsafe_allow_html=True)

                p_pct = probs["PNEUMONIA"] * 100
                n_pct = probs["NORMAL"]    * 100
                st.markdown(f"""
                <div class='prob-section'>
                    <div class='prob-row'>
                        <div class='prob-label'>
                            <span>Pneumonia</span>
                            <span class='prob-pct' style='color:#dd2222;'>{p_pct:.1f}%</span>
                        </div>
                        <div class='prob-bar-bg'>
                            <div class='prob-bar-fill-p' style='width:{p_pct:.1f}%;'></div>
                        </div>
                    </div>
                    <div class='prob-row'>
                        <div class='prob-label'>
                            <span>Normal</span>
                            <span class='prob-pct' style='color:#117733;'>{n_pct:.1f}%</span>
                        </div>
                        <div class='prob-bar-bg'>
                            <div class='prob-bar-fill-n' style='width:{n_pct:.1f}%;'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if first.strip() and last.strip() and email.strip():
                    pdf_data = build_pdf(
                        first.strip(), last.strip(), email.strip(),
                        label, conf, probs, img_pil, bundle, rec_items
                    )
                    fname = f"PneumoScan_{last.strip()}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        "⬇  Download PDF Report",
                        data=pdf_data, file_name=fname, mime="application/pdf"
                    )
                else:
                    st.markdown(
                        f"<div class='info-hint'>ℹ Fill in patient details above to enable PDF report download.</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            recs_html = "".join([
                f"<div class='rec-item'><div class='rec-dot'></div><div>{item}</div></div>"
                for item in rec_items
            ])
            st.markdown(f"""
            <div class='rec-panel {rec_type}'>
                <h4>{rec_title}</h4>
                {recs_html}
            </div>
            """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════
elif page == "About":
    st.markdown(f"""
    <div class='page-hero'>
        <div class='hero-badge'><div class='hero-dot'></div>Project Documentation</div>
        <h1>About This <em>Project</em></h1>
        <p>Background, methodology, dataset information, and academic references.</p>
    </div>
    <div class='hero-rule'></div><br>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Project Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='about-block'>
        <h4>Development of a PUMA-Optimized CNN for Pneumonia Detection from Chest X-Ray</h4>
        <p>This system implements a hybrid architecture where a Convolutional Neural Network is
        automatically tuned by the PUMA (Predator-Prey Metaheuristic) algorithm across hyperparameter
        dimensions including learning rate, dropout, filter sizes, dense units, and batch size.
        The model classifies chest X-ray images as pneumonia-positive or normal.</p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("<div class='section-label'>Dataset</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='about-block'>
            <h4>Chest X-Ray Images (Pneumonia) — Kaggle</h4>
            <p>5,863 JPEG chest X-ray images in Normal and Pneumonia categories. Published by
            Paul Mooney (2018) under CC BY 4.0. Source institution: Guangzhou Women and
            Children's Medical Center. All images validated by expert physicians.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Architecture</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='about-block'>
            <h4>CNN + PUMA Hybrid Model</h4>
            <p>Stacked convolutional blocks with batch normalization, max pooling, and dropout,
            followed by global average pooling and a dense sigmoid output. PUMA evaluates
            candidate hyperparameter sets using validation accuracy as the fitness function.</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("<div class='section-label'>Training Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='about-block'>
            <h4>Preprocessing and Augmentation</h4>
            <p>Images are resized to 224×224 and normalized. Training applies rotation,
            horizontal flip, zoom, and shear augmentation. Final training uses early stopping
            and ReduceLROnPlateau callbacks.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Evaluation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='about-block'>
            <h4>Performance Metrics</h4>
            <p>The PUMA-CNN is evaluated on a held-out test set using accuracy, precision,
            recall, and F1 score, and benchmarked against a standard Adam-trained CNN baseline.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center;padding:1.5rem 0;color:{footer_color};
         font-size:0.8rem;line-height:2.2;font-weight:400;'>
        Department of Computer Science &amp; Information Technology<br>
        College of Natural &amp; Applied Sciences, Bells University of Technology, Ota<br>
        <span style='font-size:0.72rem;'>Submitted in partial fulfillment of the requirements
        for the award of B.Tech (Hons) in Computer Science</span>
    </div>
    """, unsafe_allow_html=True)
