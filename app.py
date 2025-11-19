import streamlit as st
from PIL import Image, ImageOps
import io, base64, os, shutil, tempfile, re
import numpy as np
import cv2
import easyocr
from zipfile import ZipFile
import pandas as pd

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="📦 Asset Renaming", layout="centered")

# OCR Reader (EasyOCR, aman untuk Streamlit Cloud)
reader = easyocr.Reader(['en'], gpu=False)

# --------------------------------
# SESSION CACHE
# --------------------------------
if "cache" not in st.session_state:
    st.session_state.cache = {}   # filename → preview, rotated, ocr_text


# --------------------------------
# UTIL FUNCTIONS
# --------------------------------
def pil_to_bytes(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=85)
    return buf.getvalue()

def pil_to_b64_url(pil_img):
    return "data:image/jpeg;base64," + base64.b64encode(pil_to_bytes(pil_img)).decode()

def rotate_candidates_cv(image_cv):
    rots = []
    for angle in (0, 90, 180, 270):
        if angle == 0:
            rots.append((0, image_cv.copy()))
        else:
            h, w = image_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image_cv, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            rots.append((angle, rotated))
    return rots

def ocr_extract_text(pil_img):
    arr = np.array(pil_img)
    result = reader.readtext(arr, detail=0)
    return " ".join(result)

def find_best_rotation(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    best = {"score": -1, "pil": pil_img, "ocr": ""}

    for angle, rot in rotate_candidates_cv(cv_img):
        pil_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
        txt = ocr_extract_text(pil_rot)
        score = len(re.findall(r'\d', txt))

        if score > best["score"]:
            best = {"score": score, "pil": pil_rot, "ocr": txt}

    return best

def extract_barcode(text):
    m = re.search(r"(24\d{6,8})", text)
    return m.group(1) if m else None


# --------------------------------
# UI
# --------------------------------
st.title("📦 Asset Renaming")

st.markdown("""
Upload foto Barcode, SN, Asset → sistem otomatis membaca teks lalu rename file.

### Fitur:
- Auto rotate (0°, 90°, 180°, 270°)
- OCR-only (EasyOCR)
- Reuse foto asset
- Preview rapi 240×240
- Export ZIP hasil rename
""")

total_assets = st.number_input("Jumlah Asset:", 1, 50, 1)

asset_inputs = []
st.markdown("---")


# --------------------------------
# MAIN LOOP
# --------------------------------
for idx in range(total_assets):

    st.subheader(f"📁 Asset {idx+1}")

    uploaded = st.file_uploader(
        f"Upload foto Barcode, SN, Asset (3 foto) untuk Asset {idx+1}",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"u_{idx}"
    )

    if not uploaded:
        asset_inputs.append(None)
        continue

    uploaded = uploaded[:3]

    st.markdown("### 🖼 Preview Foto:")
    cols = st.columns(len(uploaded))
    role_map = {}

    for i, file in enumerate(uploaded):

        # cache process
        if file.name not in st.session_state.cache:
            raw = file.read()
            file.seek(0)
            pil = Image.open(io.BytesIO(raw)).convert("RGB")

            best = find_best_rotation(pil)
            rotated = best["pil"]
            ocr_txt = best["ocr"]

            thumb = rotated.copy()
            thumb.thumbnail((240,240))
            preview_url = pil_to_b64_url(thumb)

            st.session_state.cache[file.name] = {
                "preview": preview_url,
                "rotated": rotated,
                "ocr": ocr_txt
            }

        cache = st.session_state.cache[file.name]

        with cols[i]:
            st.markdown(
                f"""
                <div style='width:240px; height:240px; overflow:hidden;
                border-radius:12px; border:1px solid #ddd;'>
                    <img src="{cache['preview']}" style="width:100%; height:100%; object-fit:cover;">
                </div>
                """,
                unsafe_allow_html=True
            )

            role = st.selectbox(
                f"Nama foto ke-{i+1}",
                ["Barcode", "SN", "Asset"],
                key=f"role_{idx}_{i}"
            )
            role_map[file.name] = role

    if len(set(role_map.values())) < len(uploaded):
        st.warning("⚠ Setiap foto harus unik (Barcode, SN, Asset).")
        asset_inputs.append(None)
        continue

    asset_inputs.append({"role_map": role_map})


# --------------------------------
# PROCESS BUTTON
# --------------------------------
st.markdown("---")

if st.button("🚀 PROSES & DOWNLOAD ZIP"):

    incomplete = [i+1 for i,a in enumerate(asset_inputs) if a is None]
    if incomplete:
        st.error(f"Asset berikut belum lengkap: {incomplete}")
        st.stop()

    tmp = tempfile.mkdtemp(prefix="rename_")
    outdir = os.path.join(tmp, "output")
    os.makedirs(outdir, exist_ok=True)

    summary = []

    for idx, asset in enumerate(asset_inputs):
        role_map = asset["role_map"]

        # Barcode
        barcode_file = next(fn for fn,r in role_map.items() if r=="Barcode")
        cache = st.session_state.cache[barcode_file]

        barcode_text = cache["ocr"]
        barcode_val = extract_barcode(barcode_text)

        if not barcode_val:
            barcode_val = f"ASSET{idx+1}"

        # Save
        cache["rotated"].save(os.path.join(outdir, f"Barcode_{barcode_val}.jpg"))

        sn_file = next(fn for fn,r in role_map.items() if r=="SN")
        st.session_state.cache[sn_file]["rotated"].save(
            os.path.join(outdir, f"SN_{barcode_val}.jpg")
        )

        asset_file = next(fn for fn,r in role_map.items() if r=="Asset")
        st.session_state.cache[asset_file]["rotated"].save(
            os.path.join(outdir, f"Asset_{barcode_val}.jpg")
        )

        summary.append({
            "asset": idx+1,
            "barcode": barcode_val
        })

    # CSV
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # ZIP
    zip_path = os.path.join(tmp, "hasil_rename.zip")
    with ZipFile(zip_path, "w") as z:
        for f in os.listdir(outdir):
            z.write(os.path.join(outdir, f), arcname=f)

    st.success("Selesai! File siap di-download.")
    with open(zip_path, "rb") as f:
        st.download_button("📦 Download ZIP", f, "hasil_rename.zip")
