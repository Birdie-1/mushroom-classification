import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from streamlit_option_menu import option_menu

# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide",
)

# ============================================================
# Custom CSS — Premium Styling
# ============================================================
st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Root Variables — Forest Green + Gold ---------- */
:root {
    --bg-primary: #0f1f15;
    --bg-card: #162b1f;
    --bg-card-hover: #1f3a2b;
    --accent-gold: #d4a843;
    --accent-gold-light: #f0d078;
    --accent-emerald: #2ecc71;
    --accent-forest: #1a7a4a;
    --accent-red: #e74c3c;
    --accent-amber: #e6a117;
    --accent-teal: #1abc9c;
    --text-primary: #d4a843;
    --text-secondary: #8a9a8e;
    --border-color: #1e3328;
    --gradient-main: linear-gradient(135deg, #1a7a4a 0%, #2ecc71 100%);
    --gradient-gold: linear-gradient(135deg, #d4a843 0%, #f0d078 100%);
    --gradient-red: linear-gradient(135deg, #e74c3c 0%, #ff6b6b 100%);
}

/* ---------- Global ---------- */
html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stApp"] {
    background: linear-gradient(180deg, #0f1f15 0%, #132818 50%, #0f1f15 100%);
}

/* ---------- Hide Sidebar ---------- */
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* ---------- Hide Default Header/Footer ---------- */
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ---------- Main Container ---------- */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ---------- Hero Title ---------- */
.hero-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.hero-title h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #d4a843 0%, #f0d078 40%, #2ecc71 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-title p {
    color: var(--accent-gold);
    font-size: 0.95rem;
    font-weight: 300;
}

/* ---------- Metric Card ---------- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}
.metric-card .metric-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--accent-gold);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* Color variants — Forest theme */
.metric-card.gold::before    { background: var(--gradient-gold); }
.metric-card.emerald::before { background: var(--gradient-main); }
.metric-card.forest::before  { background: linear-gradient(135deg, #1a7a4a 0%, #145a38 100%); }
.metric-card.amber::before   { background: linear-gradient(135deg, #e6a117 0%, #c78c0e 100%); }
.metric-card.teal::before    { background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%); }
.metric-card.red::before     { background: var(--gradient-red); }
.metric-card.gold .metric-value    { color: #d4a843; }
.metric-card.emerald .metric-value { color: #2ecc71; }
.metric-card.forest .metric-value  { color: #27ae60; }
.metric-card.amber .metric-value   { color: #e6a117; }
.metric-card.teal .metric-value    { color: #1abc9c; }
.metric-card.red .metric-value     { color: #e74c3c; }

/* ---------- Result Cards ---------- */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeInUp 0.5s ease;
}
.result-edible {
    background: linear-gradient(135deg, rgba(46,204,113,0.15) 0%, rgba(26,122,74,0.08) 100%);
    border: 1px solid rgba(46,204,113,0.3);
}
.result-poisonous {
    background: linear-gradient(135deg, rgba(231,76,60,0.15) 0%, rgba(255,107,107,0.08) 100%);
    border: 1px solid rgba(231,76,60,0.3);
}
.result-card .result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-card .result-text {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-edible .result-text { color: #2ecc71; }
.result-poisonous .result-text { color: #e74c3c; }
.result-card .result-confidence {
    font-size: 0.95rem;
    color: var(--text-secondary);
}

/* ---------- Global Headings → Gold ---------- */
h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: var(--accent-gold) !important;
}

/* ---------- Section Card ---------- */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.section-card h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 1rem;
}

/* ---------- Selectbox Styling ---------- */
[data-testid="stSelectbox"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---------- Button ---------- */
.stButton > button {
    background: var(--gradient-main) !important;
    color: #f0d078 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(26,122,74,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(46,204,113,0.5) !important;
}

/* ---------- Metric Widget Override ---------- */
[data-testid="stMetric"] { display: none; }

/* ---------- Animation ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- Chart Container ---------- */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# โหลดข้อมูลและเตรียมโมเดล (cache ไว้เพื่อไม่ต้องรันซ้ำ)
# ============================================================
@st.cache_data
def load_and_train():
    """โหลด dataset, encode features, train Gaussian Naive Bayes
    และคืนค่าทุกอย่างที่ต้องใช้กลับออกมา"""

    # --- โหลดข้อมูล ---
    df = pd.read_csv("mushrooms.csv")

    # --- ตัด veil-type ออก (มีค่าเดียว ไม่มีผลต่อการทำนาย) ---
    df = df.drop(columns=["veil-type"])

    # --- เก็บ mapping ของแต่ละคอลัมน์ (ใช้แสดงใน selectbox) ---
    feature_cols = [c for c in df.columns if c != "class"]
    label_encoders = {}
    unique_values = {}

    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        unique_values[col] = list(le.classes_)

    # --- แยก features กับ target ---
    X = df[feature_cols]
    y = df["class"]

    # --- แบ่ง train / test ---
    features_train, features_test, labels_train, labels_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # --- สร้างและ train โมเดล ---
    model = GaussianNB()
    model.fit(features_train, labels_train)

    # --- ทำนายบน test set ---
    predicted_labels = model.predict(features_test)
    predicted_proba = model.predict_proba(features_test)[:, 1]

    # --- คำนวณ metrics ---
    metrics = {
        "Accuracy": accuracy_score(labels_test, predicted_labels),
        "Precision": precision_score(labels_test, predicted_labels),
        "Recall": recall_score(labels_test, predicted_labels),
        "F1-Score": f1_score(labels_test, predicted_labels),
        "MCC": matthews_corrcoef(labels_test, predicted_labels),
        "AUC-ROC": roc_auc_score(labels_test, predicted_proba),
    }

    # --- คำนวณ confusion matrix ---
    cm = confusion_matrix(labels_test, predicted_labels)

    # --- คำนวณ ROC curve ---
    fpr, tpr, _ = roc_curve(labels_test, predicted_proba)

    return (
        model, label_encoders, unique_values,
        feature_cols, metrics, cm, fpr, tpr,
    )


# --- เรียกฟังก์ชันโหลดและ train ---
(
    model, label_encoders, unique_values,
    feature_cols, metrics, cm, fpr, tpr,
) = load_and_train()

# ============================================================
# ชื่อ feature แบบอ่านง่าย
# ============================================================
FEATURE_LABELS = {
    "cap-shape": "Cap Shape (รูปทรงหมวก)",
    "cap-surface": "Cap Surface (ผิวหมวก)",
    "cap-color": "Cap Color (สีหมวก)",
    "bruises": "Bruises (รอยช้ำ)",
    "odor": "Odor (กลิ่น)",
    "gill-attachment": "Gill Attachment (การเกาะของครีบ)",
    "gill-spacing": "Gill Spacing (ระยะห่างครีบ)",
    "gill-size": "Gill Size (ขนาดครีบ)",
    "gill-color": "Gill Color (สีครีบ)",
    "stalk-shape": "Stalk Shape (รูปทรงก้าน)",
    "stalk-root": "Stalk Root (รากก้าน)",
    "stalk-surface-above-ring": "Stalk Surface Above Ring (ผิวก้านเหนือวง)",
    "stalk-surface-below-ring": "Stalk Surface Below Ring (ผิวก้านใต้วง)",
    "stalk-color-above-ring": "Stalk Color Above Ring (สีก้านเหนือวง)",
    "stalk-color-below-ring": "Stalk Color Below Ring (สีก้านใต้วง)",
    "veil-color": "Veil Color (สีเยื่อ)",
    "ring-number": "Ring Number (จำนวนวง)",
    "ring-type": "Ring Type (ชนิดวง)",
    "spore-print-color": "Spore Print Color (สีสปอร์)",
    "population": "Population (ประชากร)",
    "habitat": "Habitat (ถิ่นอาศัย)",
}

# ============================================================
# คำอธิบายค่าของแต่ละ feature (จาก UCI Mushroom Dataset)
# ============================================================
VALUE_DESCRIPTIONS = {
    "cap-shape": {
        "b": "Bell (ระฆัง)", "c": "Conical (กรวย)", "f": "Flat (แบน)",
        "k": "Knobbed (ปุ่ม)", "s": "Sunken (บุ๋ม)", "x": "Convex (นูน)",
    },
    "cap-surface": {
        "f": "Fibrous (เส้นใย)", "g": "Grooves (ร่อง)",
        "s": "Smooth (เรียบ)", "y": "Scaly (เกล็ด)",
    },
    "cap-color": {
        "b": "Buff (น้ำตาลอ่อน)", "c": "Cinnamon (อบเชย)", "e": "Red (แดง)",
        "g": "Gray (เทา)", "n": "Brown (น้ำตาล)", "p": "Pink (ชมพู)",
        "r": "Green (เขียว)", "u": "Purple (ม่วง)",
        "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "bruises": {
        "f": "No (ไม่มี)", "t": "Yes (มี)",
    },
    "odor": {
        "a": "Almond (อัลมอนด์)", "c": "Creosote (ครีโอโซต)",
        "f": "Foul (เหม็น)", "l": "Anise (โป๊ยกั๊ก)",
        "m": "Musty (อับ)", "n": "None (ไม่มีกลิ่น)",
        "p": "Pungent (ฉุน)", "s": "Spicy (เผ็ด)", "y": "Fishy (คาว)",
    },
    "gill-attachment": {
        "a": "Attached (ติด)", "f": "Free (อิสระ)",
    },
    "gill-spacing": {
        "c": "Close (ชิด)", "w": "Crowded (แน่น)",
    },
    "gill-size": {
        "b": "Broad (กว้าง)", "n": "Narrow (แคบ)",
    },
    "gill-color": {
        "b": "Buff (น้ำตาลอ่อน)", "e": "Red (แดง)",
        "g": "Gray (เทา)", "h": "Chocolate (ช็อกโกแลต)",
        "k": "Black (ดำ)", "n": "Brown (น้ำตาล)",
        "o": "Orange (ส้ม)", "p": "Pink (ชมพู)",
        "r": "Green (เขียว)", "u": "Purple (ม่วง)",
        "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "stalk-shape": {
        "e": "Enlarging (ขยาย)", "t": "Tapering (เรียว)",
    },
    "stalk-root": {
        "?": "Missing (ไม่ทราบ)", "b": "Bulbous (หัว)",
        "c": "Club (กระบอง)", "e": "Equal (เท่ากัน)", "r": "Rooted (มีราก)",
    },
    "stalk-surface-above-ring": {
        "f": "Fibrous (เส้นใย)", "k": "Silky (ไหม)",
        "s": "Smooth (เรียบ)", "y": "Scaly (เกล็ด)",
    },
    "stalk-surface-below-ring": {
        "f": "Fibrous (เส้นใย)", "k": "Silky (ไหม)",
        "s": "Smooth (เรียบ)", "y": "Scaly (เกล็ด)",
    },
    "stalk-color-above-ring": {
        "b": "Buff (น้ำตาลอ่อน)", "c": "Cinnamon (อบเชย)",
        "e": "Red (แดง)", "g": "Gray (เทา)", "n": "Brown (น้ำตาล)",
        "o": "Orange (ส้ม)", "p": "Pink (ชมพู)",
        "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "stalk-color-below-ring": {
        "b": "Buff (น้ำตาลอ่อน)", "c": "Cinnamon (อบเชย)",
        "e": "Red (แดง)", "g": "Gray (เทา)", "n": "Brown (น้ำตาล)",
        "o": "Orange (ส้ม)", "p": "Pink (ชมพู)",
        "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "veil-color": {
        "n": "Brown (น้ำตาล)", "o": "Orange (ส้ม)",
        "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "ring-number": {
        "n": "None (ไม่มี)", "o": "One (1 วง)", "t": "Two (2 วง)",
    },
    "ring-type": {
        "e": "Evanescent (จางหาย)", "f": "Flaring (บาน)",
        "l": "Large (ใหญ่)", "n": "None (ไม่มี)", "p": "Pendant (ห้อย)",
    },
    "spore-print-color": {
        "b": "Buff (น้ำตาลอ่อน)", "h": "Chocolate (ช็อกโกแลต)",
        "k": "Black (ดำ)", "n": "Brown (น้ำตาล)",
        "o": "Orange (ส้ม)", "r": "Green (เขียว)",
        "u": "Purple (ม่วง)", "w": "White (ขาว)", "y": "Yellow (เหลือง)",
    },
    "population": {
        "a": "Abundant (อุดมสมบูรณ์)", "c": "Clustered (เป็นกลุ่ม)",
        "n": "Numerous (จำนวนมาก)", "s": "Scattered (กระจาย)",
        "v": "Several (หลายตัว)", "y": "Solitary (โดดเดี่ยว)",
    },
    "habitat": {
        "d": "Woods (ป่า)", "g": "Grasses (ทุ่งหญ้า)",
        "l": "Leaves (ใบไม้)", "m": "Meadows (ทุ่งเลี้ยงสัตว์)",
        "p": "Paths (ทางเดิน)", "u": "Urban (เมือง)", "w": "Waste (ที่รกร้าง)",
    },
}

# ============================================================
# Hero Title
# ============================================================
st.markdown("""
<div class="hero-title">
    <h1>🍄 Mushroom Classification</h1>
    <p>ระบบจำแนกเห็ดด้วย Gaussian Naive Bayes — กินได้หรือมีพิษ?</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Horizontal Menu (แทน Sidebar)
# ============================================================
page = option_menu(
    menu_title=None,
    options=["ทำนายเห็ด", "ประสิทธิภาพโมเดล", "Confusion Matrix", "ROC Curve"],
    icons=["search", "bar-chart-fill", "grid-3x3-gap-fill", "graph-up"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0.5rem 0",
            "background-color": "rgba(19,31,24,0.85)",
            "border-radius": "16px",
            "border": "1px solid #1e3328",
            "backdrop-filter": "blur(10px)",
            "margin-bottom": "1.5rem",
        },
        "icon": {
            "color": "#8a9a8e",
            "font-size": "1rem",
        },
        "nav-link": {
            "font-size": "0.9rem",
            "font-weight": "500",
            "color": "#8a9a8e",
            "border-radius": "12px",
            "padding": "0.6rem 1.2rem",
            "margin": "0 0.2rem",
            "transition": "all 0.3s ease",
            "--hover-color": "#1a2e22",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #1a7a4a 0%, #2ecc71 100%)",
            "color": "#f0d078",
            "font-weight": "600",
            "box-shadow": "0 4px 15px rgba(26,122,74,0.4)",
        },
    },
)

# ============================================================
# Helper: สร้าง Matplotlib figure แบบ dark theme
# ============================================================
def create_dark_fig(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#162b1f')
    ax.set_facecolor('#162b1f')
    ax.tick_params(colors='#8a9a8e')
    ax.xaxis.label.set_color('#d4a843')
    ax.yaxis.label.set_color('#d4a843')
    ax.title.set_color('#d4a843')
    for spine in ax.spines.values():
        spine.set_color('#1e3328')
    return fig, ax

# ============================================================
# หน้า 1 — ทำนายเห็ด
# ============================================================
if page == "ทำนายเห็ด":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### เลือกคุณสมบัติของเห็ด")
    st.markdown(
        '<p style="color:#8a9a8e;font-size:0.9rem;margin-bottom:1.2rem;">'
        'กรอกคุณสมบัติทั้ง 21 รายการแล้วกดปุ่ม <b>ทำนาย</b></p>',
        unsafe_allow_html=True,
    )

    # สร้าง selectbox สำหรับแต่ละ feature (แบ่ง 3 คอลัมน์)
    # แสดงคำอธิบายให้ผู้ใช้เข้าใจเช่น "b — Bell (ระฆัง)"
    input_values = {}  # เก็บค่า raw code ที่ผู้ใช้เลือก
    cols = st.columns(3)
    for idx, col_name in enumerate(feature_cols):
        with cols[idx % 3]:
            label = FEATURE_LABELS.get(col_name, col_name)
            raw_options = unique_values[col_name]
            desc_map = VALUE_DESCRIPTIONS.get(col_name, {})

            # สร้าง display options เช่น "b — Bell (ระฆัง)"
            display_options = [
                f"{v} — {desc_map[v]}" if v in desc_map else v
                for v in raw_options
            ]

            selected_display = st.selectbox(label, display_options, key=col_name)
            # ดึงค่า raw code กลับมา (ตัวอักษรก่อน " — ")
            raw_code = selected_display.split(" — ")[0]
            input_values[col_name] = raw_code

    st.markdown("</div>", unsafe_allow_html=True)

    # ปุ่มทำนาย
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        predict_clicked = st.button("ทำนาย", use_container_width=True)

    if predict_clicked:
        # encode ค่าที่ผู้ใช้เลือก
        encoded = []
        for col_name in feature_cols:
            le = label_encoders[col_name]
            encoded.append(le.transform([input_values[col_name]])[0])

        input_array = np.array(encoded).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0]

        # แปลผลลัพธ์
        class_le = label_encoders["class"]
        predicted_label = class_le.inverse_transform([prediction])[0]
        confidence = proba[prediction]

        if predicted_label == "e":
            st.markdown(f"""
            <div class="result-card result-edible">
                <div class="result-emoji">✅</div>
                <div class="result-text">กินได้ (Edible)</div>
                <div class="result-confidence">ความมั่นใจ: {confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-poisonous">
                <div class="result-emoji">☠️</div>
                <div class="result-text">มีพิษ (Poisonous)</div>
                <div class="result-confidence">ความมั่นใจ: {confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# หน้า 2 — ประสิทธิภาพโมเดล
# ============================================================
elif page == "ประสิทธิภาพโมเดล":
    st.markdown("### ประสิทธิภาพของโมเดล Gaussian Naive Bayes")
    st.markdown(
        '<p style="color:#8a9a8e;font-size:0.9rem;margin-bottom:1.5rem;">'
        'ผลการประเมินโมเดลบน Test Set (20% ของข้อมูลทั้งหมด)</p>',
        unsafe_allow_html=True,
    )

    # Metric card data: (name, value, icon, color_class)
    card_data = [
        ("Accuracy",  metrics["Accuracy"],  "", "gold"),
        ("Precision", metrics["Precision"], "", "emerald"),
        ("Recall",    metrics["Recall"],    "", "forest"),
        ("F1-Score",  metrics["F1-Score"],  "", "amber"),
        ("MCC",       metrics["MCC"],       "", "teal"),
        ("AUC-ROC",   metrics["AUC-ROC"],   "", "red"),
    ]

    # แสดง 3 คอลัมน์ x 2 แถว
    for row_start in range(0, 6, 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            name, value, icon, color = card_data[idx]
            with col:
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
        # เพิ่มช่องว่างระหว่างแถว
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Performance Radar + Metric Guide ---
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col_radar, col_guide = st.columns([3, 2])

    with col_radar:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Radar chart
        radar_labels = list(metrics.keys())
        radar_values = list(metrics.values())

        # ปิด polygon โดยเพิ่มค่าแรกต่อท้าย
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values_plot = radar_values + [radar_values[0]]
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#162b1f')
        ax.set_facecolor('#162b1f')

        # วาด radar
        ax.plot(angles, radar_values_plot, 'o-', linewidth=2.5, color='#2ecc71', markersize=6)
        ax.fill(angles, radar_values_plot, alpha=0.15, color='#2ecc71')

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=11, fontweight='600', color='#d4a843')

        # Grid styling
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8, color='#8a9a8e')
        ax.spines['polar'].set_color('#1e3328')
        ax.grid(color='#1e3328', linewidth=0.8)
        ax.tick_params(colors='#8a9a8e')
        ax.set_title('Performance Radar', fontsize=16, fontweight='700',
                      color='#d4a843', pad=20)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_guide:
        metric_guide = {
            "Accuracy": "สัดส่วนการทำนายที่ถูกต้องทั้งหมด",
            "Precision": "สัดส่วนที่ทำนายว่า Positive แล้วถูกจริง",
            "Recall": "สัดส่วนที่เป็น Positive จริงแล้วทำนายถูก",
            "F1-Score": "ค่าเฉลี่ยฮาร์โมนิกของ Precision & Recall",
            "AUC-ROC": "พื้นที่ใต้เส้น ROC (ยิ่งใกล้ 1 ยิ่งดี)",
            "MCC": "Matthews Correlation Coefficient (-1 ถึง 1)",
        }

        st.markdown('<div class="section-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("### Metric Guide")
        for name, desc in metric_guide.items():
            st.markdown(f"""
            <div style="margin-bottom:1rem;padding-bottom:0.8rem;border-bottom:1px solid #1e3328;">
                <div style="font-weight:700;color:#d4a843;font-size:0.95rem;">{name}</div>
                <div style="color:#8a9a8e;font-size:0.85rem;margin-top:0.2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# หน้า 3 — Confusion Matrix
# ============================================================
elif page == "Confusion Matrix":
    st.markdown("### Confusion Matrix")
    st.markdown(
        '<p style="color:#8a9a8e;font-size:0.9rem;margin-bottom:1rem;">'
        'แสดงจำนวนการทำนายถูก/ผิดของแต่ละ class</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    fig, ax = create_dark_fig(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=["Edible", "Poisonous"],
        yticklabels=["Edible", "Poisonous"],
        ax=ax,
        annot_kws={"size": 18, "weight": "bold"},
        linewidths=2,
        linecolor="#162b1f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=13, fontweight="600", labelpad=12)
    ax.set_ylabel("Actual", fontsize=13, fontweight="600", labelpad=12)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="700", pad=16)
    ax.tick_params(labelsize=12)

    # colorbar text color
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#8a9a8e')

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # สรุปตัวเลข
    tn, fp, fn, tp = cm.ravel()
    col1, col2, col3, col4 = st.columns(4)
    summary_items = [
        ("True Positive", tp, "emerald"),
        ("True Negative", tn, "gold"),
        ("False Positive", fp, "amber"),
        ("False Negative", fn, "red"),
    ]
    for col, (label, val, color) in zip([col1, col2, col3, col4], summary_items):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}" style="margin-top:1rem;">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.6rem;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# หน้า 4 — ROC Curve
# ============================================================
elif page == "ROC Curve":
    st.markdown("### ROC Curve")
    st.markdown(
        '<p style="color:#8a9a8e;font-size:0.9rem;margin-bottom:1rem;">'
        'กราฟแสดงความสัมพันธ์ระหว่าง True Positive Rate กับ False Positive Rate</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    fig, ax = create_dark_fig(figsize=(8, 5.5))

    # ROC Curve
    ax.plot(
        fpr, tpr,
        color="#2ecc71", lw=2.5,
        label=f"ROC Curve (AUC = {metrics['AUC-ROC']:.4f})",
    )

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2ecc71")

    # Random line
    ax.plot(
        [0, 1], [0, 1],
        color="#d4a843", lw=1.5, linestyle="--", alpha=0.6,
        label="Random Classifier",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="600", labelpad=12)
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="600", labelpad=12)
    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=15, fontweight="700", pad=16,
    )
    ax.legend(
        loc="lower right", fontsize=11,
        facecolor="#162b1f", edgecolor="#1e3328",
        labelcolor="#d4a843",
    )
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.1, color="#8a9a8e")

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # AUC highlight card
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"""
        <div class="metric-card gold" style="margin-top:0.5rem;">
            <div class="metric-icon">🏆</div>
            <div class="metric-label">Area Under Curve (AUC)</div>
            <div class="metric-value">{metrics['AUC-ROC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
