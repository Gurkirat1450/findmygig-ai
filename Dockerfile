# ---------------------------------------------------------------------
# Stage 1: builder — installs everything needed to INSTALL dependencies
# (pip's cache, wheel build artifacts, torch's CPU index, etc). None of
# this needs to exist in the final image; it's just scaffolding.
# ---------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

# --user installs into /root/.local instead of system site-packages —
# that's what makes it possible to copy ONLY the installed packages into
# the runtime stage below, leaving pip's download cache and any build
# artifacts behind in this (discarded) builder stage.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------------------------
# Stage 2: runtime — starts fresh from a clean slim image, copies ONLY
# the installed packages (not pip itself, not the cache, not any
# intermediate files from the builder stage). This is the actual size
# saving: compare `docker images` before/after this change.
# ---------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8000

# Self-contained healthcheck — uses Python's stdlib (urllib) instead of
# curl, since curl isn't installed on python:slim by default and adding
# it would defeat the point of keeping this image lean.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
