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
GPU_LAYERS="${OCR_GPU_LAYERS:-99}"
REQUIRE_GPU="${OCR_REQUIRE_GPU:-0}"
API_KEY="${OCR_API_KEY:-}"

# ROCm: un HSA_OVERRIDE_GFX_VERSION definido pero VACÍO hace que HIP no detecte
# ninguna GPU y llama.cpp caiga silenciosamente a CPU. Tratar vacío como
# "sin override" (detección nativa) para evitar ese fallback silencioso.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
  unset HSA_OVERRIDE_GFX_VERSION
fi

# verify_gpu: con OCR_REQUIRE_GPU=1 aborta si llama.cpp no ve una GPU real. Evita
# que el contenedor se marque saludable ejecutando en CPU (o entre en bucle de
# reinicio sin un mensaje claro).
verify_gpu() {
  if [ "${REQUIRE_GPU}" != "1" ]; then
    return 0
  fi
  local devices
  devices="$(llama-server --list-devices 2>&1 || true)"
  # Ancla a las líneas de dispositivo reales ("  ROCm0: ...", "  CUDA0: ...") para
  # no dar falso positivo con mensajes de error que mencionan "ROCm".
  if ! printf '%s\n' "${devices}" | grep -Eq '^[[:space:]]*(ROCm|CUDA)[0-9]+:'; then
    echo "ERROR: OCR_REQUIRE_GPU=1 pero no se detectó ninguna GPU." >&2
    printf '%s\n' "${devices}" >&2
    echo "Revisa driver/ROCm, el acceso a /dev/kfd y /dev/dri, y que" >&2
    echo "HSA_OVERRIDE_GFX_VERSION no esté definido vacío." >&2
    return 1
  fi
  echo "GPU detectada:"
  printf '%s\n' "${devices}" | grep -E '^[[:space:]]*(ROCm|CUDA)[0-9]+:' | sed 's/^/  /'
}

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

# Verifica (si se exige) que haya GPU antes de arrancar; falla con mensaje claro
# en lugar de degradar a CPU.
verify_gpu || exit 1

args=(
  --host 0.0.0.0
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --ctx-size "${CONTEXT}"
  --n-gpu-layers "${GPU_LAYERS}"
  --image-min-tokens 1024
  --parallel "${PARALLEL}"
  --threads "${THREADS}"
)

if [ -n "${API_KEY}" ]; then
  args+=(--api-key "${API_KEY}")
fi

if [ -n "${MMPROJ_PATH}" ]; then
  args+=(--mmproj "${MMPROJ_PATH}")
fi

echo "Iniciando OCR: modelo=${MODEL_PATH} mmproj=${MMPROJ_PATH:-<ninguno>} ctx=${CONTEXT} parallel=${PARALLEL} ngl=${GPU_LAYERS} puerto=${PORT} api_key=$([ -n "${API_KEY}" ] && echo 'sí' || echo 'no')"
exec llama-server "${args[@]}"
