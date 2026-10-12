#!/usr/bin/env bash
# ============================================================================
# Punto de entrada del servicio OCR local (llama.cpp `llama-server`).
# Expone una API compatible con OpenAI en OCR_PORT (por defecto 8080) y carga
# el modelo multimodal (texto + proyector visual `mmproj`) desde el volumen
# persistente ocr-models. No imprime credenciales.
# ============================================================================
set -euo pipefail

MODEL_PATH="${OCR_MODEL_PATH:-/models/model.gguf}"
MMPROJ_PATH="${OCR_MMPROJ_PATH:-}"
PORT="${OCR_PORT:-8080}"
CONTEXT="${OCR_CONTEXT_SIZE:-8192}"
PARALLEL="${OCR_CONCURRENCY:-1}"
THREADS="${OCR_THREADS:-$(nproc)}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: no se encontraron pesos del modelo en '${MODEL_PATH}'." >&2
  echo "Descarga el GGUF en el volumen 'ocr-models' o define OCR_MODEL_PATH." >&2
  exit 1
fi

args=(
  --host 0.0.0.0
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --ctx-size "${CONTEXT}"
  --parallel "${PARALLEL}"
  --threads "${THREADS}"
)

if [ -n "${MMPROJ_PATH}" ]; then
  if [ ! -f "${MMPROJ_PATH}" ]; then
    echo "ERROR: no se encontró el proyector visual en '${MMPROJ_PATH}'." >&2
    exit 1
  fi
  args+=(--mmproj "${MMPROJ_PATH}")
fi

echo "Iniciando OCR: modelo=${MODEL_PATH} ctx=${CONTEXT} parallel=${PARALLEL} puerto=${PORT}"
exec llama-server "${args[@]}"
