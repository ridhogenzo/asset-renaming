# app.py
import streamlit as st
from PIL import Image, ImageOps
import io, base64, os, shutil, tempfile, re
import numpy as np
import cv2
import easyocr
from zipfile import ZipFile
from datetime import datetime
import pandas as pd

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(page_title="Asset Renaming", layout="wide")

# ===================================
# INITIALIZE OCR (EasyOCR)
# ===================================
reader = easyocr.Reader(["en"], gpu=False)

# ===================================
# SESSION STATE CACHE
# ===================================
if "preview_cache" not in st.session_state:
    st.session_state.preview_cache = {}

if "original_img" not in st.session_state:
    st.session_state.original_img = {}

# ===================================
# STYLE (same as original)
# ===================================
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
    background: rgba(0,0,0,0.03);
}
.preview-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    cursor: pointer;
}
.select, .stSelectbox>div>div>div>select {
    width: 240px !important;
}
body { background: #0d0d0d; }
h1 { color: #f2c94c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ===================================
# LIGHTBOX VIEWER
# ===================================
st.markdown("""
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
    <img id="lb-image" style="max-width:92%; max-height:92%; border-radius:10px;">
</div>
""", unsafe_allow_html=True)

# ===================================
# HELPER FUNCTIONS
# ===================================
def pil_to_b64_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    enc = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{enc}"

def rotate_candidates_cv(cvimg):
    for angle in (0, 90, 180, 270):
        if angle == 0:
            yield cvimg
        else:
            h, w = cvimg.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            rotated = cv2.warpAffine(cvimg, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            yield rotated

def easyocr_text(pil_img):
    arr = np.array(pil_img)
    result = reader.readtext(arr, detail=0)
    return " ".join(result)

def best_preview_rotation(pil_img):
    cvimg = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    best_score = -1
    best_pil = pil_img

    for rot in rotate_candidates_cv(cvimg):
        pil_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
        txt = easyocr_text(pil_rot)
        score = len(re.findall(r"\d", txt))

        if score > best_score:
            best_score = score
            best_pil = pil_rot

    return best_pil

def extract_barcode(text):
    m = re.search(r"(24\d{6,8})", text)
    return m.group(1) if m else None

# ===================================
# PAGE TITLE
# ===================================
st.title("📦 Asset Renaming (EasyOCR Version)")
st.markdown("""
Sistem otomatis untuk merename foto Barcode, SN, dan Asset menggunakan **EasyOCR**.

- Auto rotate preview (0°, 90°, 180°, 270°)  
- FOTO ASLI tetap utuh (tidak dirotasi saat disimpan)  
- Bisa pakai Asset reuse  
- Fullscreen zoom  
""")

# ===================================
# MAIN INPUT LOOP
# ===================================
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
        f"Upload foto untuk Asset {idx+1} (max 3 file)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"u_{idx}"
    )

    if not uploaded:
        asset_records.append(None)
        continue

    uploaded = uploaded[:3]

    st.markdown("### 🖼 Preview")
    cols = st.columns(len(uploaded))
    role_map = {}

    roles = ["Barcode", "SN"] if reuse else ["Barcode", "SN", "Asset"]

    for i, f in enumerate(uploaded):
        key = f"{idx}_{f.name}"

        raw = f.read()
        f.seek(0)
        pil_orig = Image.open(io.BytesIO(raw)).convert("RGB")
        st.session_state.original_img[key] = pil_orig

        # PREVIEW ROTASI
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
                    <img src="{url}" onclick="openLightbox('{url}')"/>
                </div>
                """,
                unsafe_allow_html=True
            )
            sel = st.selectbox(
                f"Foto ke-{i+1}",
                roles,
                key=f"role_{idx}_{i}"
            )
            role_map[key] = sel

    if len(set(role_map.values())) < len(roles):
        st.warning("⚠ Barcode, SN, Asset harus unik.")
        asset_records.append(None)
        continue

    if idx == 0:
        for k, v in role_map.items():
            if v == "Asset":
                first_asset_key = k
                break

    asset_records.append({"map": role_map, "reuse": reuse})
    st.success(f"Asset {idx+1} siap.")

# ===================================
# PROCESS & DOWNLOAD
# ===================================
st.markdown("---")
if st.button("🚀 PROSES & DOWNLOAD ZIP"):

    incomplete = [i+1 for i,a in enumerate(asset_records) if a is None]
    if incomplete:
        st.error(f"Asset belum lengkap: {incomplete}")
        st.stop()

    tmp = tempfile.mkdtemp(prefix="assets_")
    outdir = os.path.join(tmp, "output")
    os.makedirs(outdir, exist_ok=True)

    summary = []

    for idx, rec in enumerate(asset_records):
        mapping = rec["map"]
        reuse = rec["reuse"]

        # BARCODE
        barcode_key = next(k for k,v in mapping.items() if v=="Barcode")
        pil_barcode = st.session_state.original_img[barcode_key]

        # DETEKSI BARCODE VIA OCR + ROTASI
        cvimg = cv2.cvtColor(np.array(pil_barcode), cv2.COLOR_RGB2BGR)
        text_all = ""
        for rot in rotate_candidates_cv(cvimg):
            pil_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
            text_all += " " + easyocr_text(pil_rot)

        barcode_val = extract_barcode(text_all)
        if not barcode_val:
            barcode_val = f"ASSET{idx+1}"

        # SAVE BARCODE
        pil_barcode.save(os.path.join(outdir, f"Barcode_{barcode_val}.jpg"))

        # SAVE SN
        sn_key = next(k for k,v in mapping.items() if v=="SN")
        st.session_state.original_img[sn_key].save(
            os.path.join(outdir, f"SN_{barcode_val}.jpg")
        )

        # SAVE ASSET
        if reuse and first_asset_key:
            st.session_state.original_img[first_asset_key].save(
                os.path.join(outdir, f"Asset_{barcode_val}.jpg")
            )
        else:
            asset_key = next(k for k,v in mapping.items() if v=="Asset")
            st.session_state.original_img[asset_key].save(
                os.path.join(outdir, f"Asset_{barcode_val}.jpg")
            )

        summary.append({"asset": idx+1, "barcode": barcode_val})

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    zip_path = os.path.join(
        tmp, f"rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )
    with ZipFile(zip_path, "w") as z:
        for f in os.listdir(outdir):
            z.write(os.path.join(outdir, f), arcname=f)

    st.success("✔ Selesai. Silakan download hasilnya:")
    with open(zip_path, "rb") as f:
        st.download_button("📦 Download ZIP", f, file_name=os.path.basename(zip_path))
