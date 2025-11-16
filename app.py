# app.py
import streamlit as st
from PIL import Image, ImageOps
import io, base64, os, shutil, tempfile, re
import numpy as np
import cv2
from pyzbar.pyzbar import decode as zbar_decode
import pytesseract
from zipfile import ZipFile
from datetime import datetime
import pandas as pd

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Asset Renaming", layout="wide")

# ----------------------------
# Session state init
# ----------------------------
if "preview_cache" not in st.session_state:
    st.session_state.preview_cache = {}   # key -> preview data url
if "original_img" not in st.session_state:
    st.session_state.original_img = {}   # key -> PIL original image (RGB)

# ----------------------------
# CSS: center container, preview size, select width
# ----------------------------
st.markdown(
    """
    <style>
    /* center main container and add side padding */
    .reportview-container .main .block-container {
        max-width: 1100px;
        padding-left: 40px;
        padding-right: 40px;
    }

    /* preview box */
    .preview-box {
        width: 240px;
        height: 240px;
        overflow: hidden;
        border-radius: 12px;
        margin-bottom: 8px;
        background: rgba(0,0,0,0.03);
    }
    .preview-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        cursor: pointer;
        display: block;
    }

    /* make dropdown/select a similar width in many browsers */
    select, .stSelectbox>div>div>div>select {
        width: 240px !important;
    }

    /* small spacing under preview */
    .preview-caption {
        font-size: 12px;
        color: #cfcfcf;
        margin-bottom: 6px;
    }
    
body {
    background: #0d0d0d;
}
h1 {
    color: #f2c94c;
    font-weight: 700;
}
.asset-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
}
.glow-button button {
    background: linear-gradient(90deg, #f2c94c, #c7971e) !important;
    color: black !important;
    font-weight: 600;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    box-shadow: 0 0 12px rgba(242,201,76,0.6);
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# LIGHTBOX HTML + JS (fullscreen + simple zoom on wheel)
# ----------------------------
st.markdown(
    """
    <script>
    function openLightbox(src){
        const m = document.getElementById("lb-modal");
        const i = document.getElementById("lb-image");
        i.src = src;
        i.style.transform = "scale(1)";
        m.style.display = "flex";
    }
    function closeLightbox(){
        document.getElementById("lb-modal").style.display = "none";
    }
    // zoom by wheel
    document.addEventListener("wheel", function(e){
        const i = document.getElementById("lb-image");
        if(!i) return;
        let s = i.style.transform.replace(/[^0-9.]/g,"") || 1;
        s = parseFloat(s);
        if(e.deltaY < 0) s += 0.1; else s -= 0.1;
        if(s < 0.2) s = 0.2;
        i.style.transform = "scale(" + s + ")";
    });
    </script>

    <div id="lb-modal" style="
        display:none; position:fixed; top:0; left:0; width:100%; height:100%;
        background:rgba(0,0,0,0.86); justify-content:center; align-items:center; z-index:9999;">
        <span onclick="closeLightbox()"
              style="position:absolute; top:22px; right:32px; font-size:34px; color:white; cursor:pointer;">&times;</span>
        <img id="lb-image" style="max-width:92%; max-height:92%; border-radius:10px; transition: transform 0.08s ease;">
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helper functions
# ----------------------------
def pil_to_b64_url(pil_img):
    """Return data URL from PIL image"""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

def rotate_candidates_cv(image_cv):
    """Yield rotated cv2 images at 0,90,180,270"""
    for angle in (0, 90, 180, 270):
        if angle == 0:
            yield image_cv
        else:
            h, w = image_cv.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(image_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            yield rotated

def detect_barcodes_in_pil(pil_img):
    """Return list of decoded barcode objects (pyzbar). Uses PIL input."""
    try:
        decs = zbar_decode(pil_img)
        results = []
        for d in decs:
            results.append({
                "data": d.data.decode(errors="ignore"),
                "rect": d.rect
            })
        return results
    except Exception:
        return []

def ocr_text_from_pil(pil_img):
    try:
        gray = ImageOps.grayscale(pil_img)
        txt = pytesseract.image_to_string(gray, lang='eng')
        return txt
    except Exception:
        return ""

def find_barcode_number_from_text(text):
    m = re.search(r"(24\d{6,8})", text)
    return m.group(1) if m else None

def best_preview_rotation(pil_img):
    """
    Choose a rotated PIL image for preview only:
    pick rotation that yields most barcode decodes (pyzbar).
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    best_score = -1
    best_pil = pil_img
    for rot_cv in rotate_candidates_cv(cv_img):
        pil_rot = Image.fromarray(cv2.cvtColor(rot_cv, cv2.COLOR_BGR2RGB))
        decs = detect_barcodes_in_pil(pil_rot)
        score = len(decs)
        # small boost from digits in OCR
        ocr = ocr_text_from_pil(pil_rot)
        score += len(re.findall(r'\d', ocr)) * 0.01
        if score > best_score:
            best_score = score
            best_pil = pil_rot
    return best_pil

# ----------------------------
# UI title/intro
# ----------------------------
st.title("📦 Asset Renaming")
st.markdown(
    """
    Sistem otomatis untuk merename foto *Barcode*, *SN*, dan *Asset* berdasarkan **nomor barcode (24xxxxxx)**.
    - Preview auto-rotate (supaya barcode terlihat jelas) — **hanya preview**
    - File hasil rename akan menggunakan **foto original** (tidak di-rotate)
    - Klik preview untuk fullscreen + zoom (wheel)
    """
)

# ----------------------------
# Main inputs
# ----------------------------
total_assets = st.number_input("Jumlah Asset:", min_value=1, max_value=50, value=1)
asset_records = []
first_asset_original_key = None

for idx in range(total_assets):
    st.markdown("---")
    st.subheader(f"📁 Asset {idx+1}")

    reuse_asset = False
    if idx > 0:
        reuse_asset = st.checkbox(f"Gunakan foto Asset dari Asset 1?", key=f"reuse_{idx}")

    uploaded = st.file_uploader(
        f"Upload maks 3 foto untuk Asset {idx+1} (Barcode, SN, Asset)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"u_{idx}"
    )

    if not uploaded:
        asset_records.append(None)
        continue

    # cap to 3 files
    uploaded = uploaded[:3]

    st.markdown("### 🖼 Preview:")
    cols = st.columns(len(uploaded))
    role_map = {}
    roles = ["Barcode", "SN"] if reuse_asset else ["Barcode", "SN", "Asset"]

    for i, f in enumerate(uploaded):
        # create unique key for this file instance (index + filename) to avoid collisions
        key_name = f"{idx}_{f.name}"

        # read original and cache it
        raw = f.read()
        f.seek(0)
        try:
            pil_orig = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pil_orig = Image.open(io.BytesIO(raw)).convert("RGB")

        st.session_state.original_img[key_name] = pil_orig

        # compute preview rotated (cache)
        if key_name not in st.session_state.preview_cache:
            preview_pil = best_preview_rotation(pil_orig)
            thumb = preview_pil.copy()
            thumb.thumbnail((240, 240))
            data_url = pil_to_b64_url(thumb)
            st.session_state.preview_cache[key_name] = data_url
        preview_url = st.session_state.preview_cache[key_name]

        # render preview + selectbox (selectbox will keep value in session)
        with cols[i]:
            st.markdown(
                f"""
                <div class="preview-box">
                    <img src="{preview_url}" onclick="openLightbox('{preview_url}')" />
                </div>
                """,
                unsafe_allow_html=True,
            )

            # if reuse and this is not first asset, don't include "Asset" in options
            sel = st.selectbox(
                f"Nama foto ke-{i+1}",
                options=roles,
                key=f"role_{idx}_{i}"
            )
            role_map[key_name] = sel

    # validate unique roles
    if len(set(role_map.values())) < len(roles):
        st.warning("⚠ Barcode, SN, Asset harus unik untuk setiap asset.")
        asset_records.append(None)
        continue

    # record first asset's asset-image key for reuse later
    if idx == 0:
        # find the key that was selected as Asset in first group
        for k, v in role_map.items():
            if v == "Asset":
                first_asset_original_key = k
                break

    asset_records.append({"mapping": role_map, "reuse": reuse_asset})
    st.success(f"✔ Asset {idx+1} siap diproses")

# ----------------------------
# Process & Download
# ----------------------------
st.markdown("---")
if st.button("🚀 PROSES & DOWNLOAD ZIP"):

    incomplete = [i+1 for i, val in enumerate(asset_records) if val is None]
    if incomplete:
        st.error(f"Asset berikut belum lengkap atau perlu diperbaiki: {incomplete}")
        st.stop()

    tmpdir = tempfile.mkdtemp(prefix="asset_rename_")
    outdir = os.path.join(tmpdir, "output")
    os.makedirs(outdir, exist_ok=True)

    summary = []

    for idx, rec in enumerate(asset_records):
        mapping = rec["mapping"]
        reuse = rec["reuse"]

        # find barcode image key
        barcode_key = next(k for k, r in mapping.items() if r == "Barcode")
        # use original PIL for saving (NOT rotated preview)
        barcode_orig = st.session_state.original_img[barcode_key]

        # try decode barcode using rotated candidates (so detection robust)
        barcode_text_all = ""
        cv_orig = cv2.cvtColor(np.array(barcode_orig), cv2.COLOR_RGB2BGR)
        for rot in rotate_candidates_cv(cv_orig):
            pil_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
            decs = detect_barcodes_in_pil(pil_rot)
            for d in decs:
                barcode_text_all += d["data"] + " "

        # fallback OCR on original
        barcode_text_all += ocr_text_from_pil(barcode_orig)

        barcode_val = find_barcode_number_from_text(barcode_text_all)
        if not barcode_val:
            barcode_val = f"ASSET{idx+1}"

        # save Barcode (original image)
        barcode_orig.save(os.path.join(outdir, f"Barcode_{barcode_val}.jpg"))

        # SN: find selected SN key
        sn_key = next(k for k, r in mapping.items() if r == "SN")
        sn_orig = st.session_state.original_img[sn_key]
        sn_orig.save(os.path.join(outdir, f"SN_{barcode_val}.jpg"))

        # Asset: if reuse -> use first_asset_original_key, else take selected key
        if "Asset" in mapping.values():
            asset_key = next(k for k, r in mapping.items() if r == "Asset")
            asset_orig = st.session_state.original_img[asset_key]
        else:
            # reuse mode
            if first_asset_original_key is None:
                # fallback: use barcode image as asset (shouldn't happen if UI forced Asset in first)
                asset_orig = barcode_orig
            else:
                asset_orig = st.session_state.original_img[first_asset_original_key]

        asset_orig.save(os.path.join(outdir, f"Asset_{barcode_val}.jpg"))

        summary.append({"asset_index": idx+1, "barcode": barcode_val})

    # write summary csv
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # create zip
    zip_path = os.path.join(tmpdir, f"asset_rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    with ZipFile(zip_path, "w") as z:
        for f in sorted(os.listdir(outdir)):
            z.write(os.path.join(outdir, f), arcname=f)

    st.success("🎉 Semua asset selesai diproses — siap diunduh.")
    with open(zip_path, "rb") as fh:
        st.download_button("📦 Download Hasil (.zip)", fh, file_name=os.path.basename(zip_path), mime="application/zip")
