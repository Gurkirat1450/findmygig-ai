FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — layer caching means this only re-runs
# when requirements.txt changes, not on every code edit.
COPY requirements.txt .

# Force the CPU-only torch build. Without this, pip resolves the default
# GPU build for sentence-transformers' torch dependency, which pulls in
# NVIDIA's CUDA toolkit + cuDNN (~900MB combined) — unnecessary for this
# project and can blow past Docker Desktop's disk allocation on Windows.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
