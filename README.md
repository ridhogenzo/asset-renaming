
# Device Scanner (Streamlit)

A simple Streamlit app that:
- Accepts multiple uploaded images (JPG/PNG)
- Detects barcode (if present) and renames images to `Barcode_<value>.jpg`
- Extracts simple OCR (SN / device name) using Tesseract
- Produces a `device_data.csv` inside the downloadable ZIP

## Step-by-step in VS Code

### 1️⃣ Install prerequisites
- Install Python 3.9+.
- Install Tesseract OCR:
  - **Windows:** download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.
  - **Linux:** `sudo apt install tesseract-ocr`
  - **macOS:** `brew install tesseract`

### 2️⃣ Open in VS Code
Extract the ZIP, then open the folder in VS Code.

### 3️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
# activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the app locally
```bash
streamlit run app.py
```

Then open the URL shown (default: http://localhost:8501).

### 6️⃣ Deploy to Streamlit Cloud
1. Push the project to GitHub.
2. Go to https://share.streamlit.io.
3. Log in → New App → select repo → choose `app.py`.
4. Wait for deployment to finish (~2–3 mins).
5. Done 🎉
