#!/usr/bin/env bash
# ============================================================================
# Punto de entrada del servicio OCR local (llama.cpp `llama-server`).
#
# Expone una API compatible con OpenAI en OCR_PORT (por defecto 8080) y carga
# el modelo multimodal (texto + proyector visual `mmproj`) desde el volumen
# persistente ocr-models.
#
# Si los pesos no existen en el volumen, se descargan automáticamente desde
# Hugging Face usando las URLs OCR_MODEL_URL / OCR_MMPROJ_URL (configurables por
# entorno). No imprime credenciales.
# ============================================================================
set -euo pipefail

# Rutas de bibliotecas compartidas: sistema, instalación local y árbol de
# compilación de llama.cpp (relevante para variantes de una sola etapa, p. ej.
# ROCm, donde las .so permanecen en /src/build/bin). Evita el error
# "libllama-server-impl.so: cannot open shared object file".
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/app/lib:/src/build/bin:${LD_LIBRARY_PATH:-}"

MODEL_PATH="${OCR_MODEL_PATH:-/models/model.gguf}"
MMPROJ_PATH="${OCR_MMPROJ_PATH:-}"
MODEL_URL="${OCR_MODEL_URL:-}"
MMPROJ_URL="${OCR_MMPROJ_URL:-}"
PORT="${OCR_PORT:-8080}"
CONTEXT="${OCR_CONTEXT_SIZE:-8192}"
PARALLEL="${OCR_CONCURRENCY:-1}"
THREADS="${OCR_THREADS:-$(nproc)}"

download_file() {
  # download_file <url> <destino>
  local url="$1"
  local dest="$2"
  local partial="${dest}.part"
  mkdir -p "$(dirname "${dest}")"
  echo "Descargando: ${url}"
  echo "         ->  ${dest}"
  if command -v curl >/dev/null 2>&1; then
    local curl_args=(-fL --retry 3 --retry-delay 5 --connect-timeout 30 --progress-bar)
    # Reanudar una descarga parcial previa si existe.
    if [ -s "${partial}" ]; then
      curl_args+=(-C -)
    fi
    curl "${curl_args[@]}" -o "${partial}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    if [ -s "${partial}" ]; then
      wget -c -O "${partial}" "${url}"
    else
      wget -O "${partial}" "${url}"
    fi
  else
    echo "ERROR: se requiere 'curl' o 'wget' para descargar los pesos." >&2
    return 1
  fi
  mv -f "${partial}" "${dest}"
}

ensure_file() {
  # ensure_file <ruta> <url> <descripcion>
  local path="$1"
  local url="$2"
  local label="$3"
  if [ -f "${path}" ]; then
    echo "${label} ya presente en '${path}'; no se descarga."
    return 0
  fi
  if [ -z "${url}" ]; then
    echo "ERROR: falta ${label} en '${path}' y no se definió su URL." >&2
    echo "Define OCR_MODEL_URL / OCR_MMPROJ_URL o monta los pesos en el volumen 'ocr-models'." >&2
    return 1
  fi
  download_file "${url}" "${path}"
  if [ ! -f "${path}" ]; then
    echo "ERROR: la descarga de ${label} falló ('${url}')." >&2
    return 1
  fi
}

ensure_file "${MODEL_PATH}" "${MODEL_URL}" "el modelo de visión" || exit 1
if [ -n "${MMPROJ_PATH}" ]; then
  ensure_file "${MMPROJ_PATH}" "${MMPROJ_URL}" "el proyector visual (mmproj)" || exit 1
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
  args+=(--mmproj "${MMPROJ_PATH}")
fi

echo "Iniciando OCR: modelo=${MODEL_PATH} mmproj=${MMPROJ_PATH:-<ninguno>} ctx=${CONTEXT} parallel=${PARALLEL} puerto=${PORT}"
exec llama-server "${args[@]}"
