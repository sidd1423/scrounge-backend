# Scrounge detection backend — FastAPI + Tesseract OCR
#
# Render/Railway/Fly all auto-detect a Dockerfile at the repo root and
# build from it. This installs the tesseract-ocr system package (the
# actual binary pytesseract wraps) alongside your Python dependencies.

FROM python:3.11-slim

# tesseract-ocr: the OCR engine binary itself (pytesseract just calls out to it)
# libgl1 / libglib2.0-0: common transitive deps for image libs (Pillow/OpenCV-style
#   wheels sometimes expect these on slim images — safe to include even if unused)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the app
COPY . .

# Render/Railway inject $PORT at runtime — bind to it instead of a hardcoded port
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
