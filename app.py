# app.py
import streamlit as st
from PIL import Image, ImageOps
import io, base64, os, tempfile, re
import numpy as np
import cv2
import easyocr
from zipfile import ZipFile
from datetime import datetime
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Asset Renaming", layout="wide")

# =========================================================
# OCR INIT (EasyOCR) — Cache agar cepat di Streamlit Cloud
# =========================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# =========================================================
# SESSION STATE
# =========================================================
if "preview_cache" not in st.session_state:
    st.session_state.preview_cache = {}

if "original_img" not in st.session_state:
    st.session_state.original_img = {}

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.reportview-container .main .block-container {
    max-width: 1100px;
    padding-left: 40px;
    padding-right: 40px;
}
.preview-box {
    width: 240px;
    height: 240px;
    overflow: hidden;
    border-radius: 12px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.05);
}
.preview-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
body { background: #0d0d0d; }
h1 { color: #f2c94c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def pil_to_b64_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def rotate_candidates_cv(cv_img):
    for angle in (0, 90, 180, 270):
        if angle == 0:
            yield cv_img
        else:
            h, w = cv_img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            yield cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def easyocr_text(pil_img):
    arr = np.array(pil_img)
    result = reader.readtext(arr, detail=0)
    return " ".join(result)

def best_preview_rotation(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    best_score = -1
    best_pil = pil_img

    for rot_cv in rotate_candidates_cv(cv_img):
        pil_rot = Image.fromarray(cv2.cvtColor(rot_cv, cv2.COLOR_BGR2RGB))
        txt = easyocr_text(pil_rot)
        score = len(re.findall(r"\d", txt))
        if score > best_score:
            best_score = score
            best_pil = pil_rot
    return best_pil

def extract_barcode(text):
    m = re.search(r"(24\d{6,8})", text)
    return m.group(1) if m else None

# =========================================================
# TITLE & DESCRIPTION
# =========================================================
st.title("📦 Asset Renaming")

st.markdown("""
Aplikasi untuk merename foto Barcode, SN, dan Asset secara otomatis.

• Auto-rotate preview  
• Gambar asli tetap utuh (tidak diubah saat disimpan)  
• Mendukung reuse foto Asset  
""")

# =========================================================
# MAIN INPUT
# =========================================================
total_assets = st.number_input("Jumlah Asset:", 1, 50, 1)
asset_records = []
first_asset_key = None

for idx in range(total_assets):
    st.markdown("---")
    st.subheader(f"📁 Asset {idx+1}")

    reuse = False
    if idx > 0:
        reuse = st.checkbox("Gunakan foto Asset dari Asset 1?", key=f"reuse_{idx}")

    uploaded = st.file_uploader(
        f"Upload foto untuk Asset {idx+1}",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"u_{idx}"
    )

    if not uploaded:
        asset_records.append(None)
        continue

    uploaded = uploaded[:3]

    roles = ["Barcode", "SN"] if reuse else ["Barcode", "SN", "Asset"]
    role_map = {}

    cols = st.columns(len(uploaded))
    st.markdown("### Preview")

    for i, f in enumerate(uploaded):
        key = f"{idx}_{f.name}"

        raw = f.read()
        f.seek(0)

        pil_orig = Image.open(io.BytesIO(raw)).convert("RGB")
        st.session_state.original_img[key] = pil_orig

        if key not in st.session_state.preview_cache:
            rot = best_preview_rotation(pil_orig)
            thumb = rot.copy()
            thumb.thumbnail((240,240))
            st.session_state.preview_cache[key] = pil_to_b64_url(thumb)

        url = st.session_state.preview_cache[key]

        with cols[i]:
            st.markdown(
                f"""
                <div class="preview-box">
                    <img src="{url}" />
                </div>
                """,
                unsafe_allow_html=True,
            )

            sel = st.selectbox(
                f"Foto ke-{i+1}",
                roles,
                key=f"role_{idx}_{i}",
            )
            role_map[key] = sel

    if len(set(role_map.values())) < len(roles):
        st.warning("⚠ Barcode, SN, Asset harus berbeda.")
        asset_records.append(None)
        continue

    if idx == 0:
        for k, v in role_map.items():
            if v == "Asset":
                first_asset_key = k
                break

    asset_records.append({"map": role_map, "reuse": reuse})

# =========================================================
# PROCESS
# =========================================================
st.markdown("---")
if st.button("🚀 PROSES & DOWNLOAD"):

    incomplete = [i+1 for i,a in enumerate(asset_records) if a is None]
    if incomplete:
        st.error(f"Asset berikut belum lengkap: {incomplete}")
        st.stop()

    tmp = tempfile.mkdtemp(prefix="assets_")
    outdir = os.path.join(tmp, "output")
    os.makedirs(outdir, exist_ok=True)

    summary = []

    for idx, rec in enumerate(asset_records):
        roles = rec["map"]
        reuse = rec["reuse"]

        # ===== Barcode =====
        barcode_key = next(k for k,v in roles.items() if v=="Barcode")
        pil_barcode = st.session_state.original_img[barcode_key]

        text_all = ""
        cvimg = cv2.cvtColor(np.array(pil_barcode), cv2.COLOR_RGB2BGR)
        for rot in rotate_candidates_cv(cvimg):
            pil_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
            text_all += " " + easyocr_text(pil_rot)

        code = extract_barcode(text_all)
        if not code:
            code = f"ASSET{idx+1}"

        pil_barcode.save(os.path.join(outdir, f"Barcode_{code}.jpg"))

        # ===== SN =====
        sn_key = next(k for k,v in roles.items() if v=="SN")
        st.session_state.original_img[sn_key].save(
            os.path.join(outdir, f"SN_{code}.jpg")
        )

        # ===== Asset =====
        if reuse and first_asset_key:
            st.session_state.original_img[first_asset_key].save(
                os.path.join(outdir, f"Asset_{code}.jpg")
            )
        else:
            a_key = next(k for k,v in roles.items() if v=="Asset")
            st.session_state.original_img[a_key].save(
                os.path.join(outdir, f"Asset_{code}.jpg")
            )

        summary.append({"asset": idx+1, "barcode": code})

    pd.DataFrame(summary).to_csv(os.path.join(outdir, "summary.csv"), index=False)

    zip_path = os.path.join(tmp, f"hasil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    with ZipFile(zip_path, "w") as z:
        for f in os.listdir(outdir):
            z.write(os.path.join(outdir, f), arcname=f)

    st.success("✔ Selesai. Silakan download hasilnya.")
    with open(zip_path, "rb") as f:
        st.download_button("📦 Download ZIP", f, file_name=os.path.basename(zip_path))
