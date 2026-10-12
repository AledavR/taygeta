# Taygeta

Banco de exámenes científicos y sistema de recuperación **MCP** con revisión
**Human-in-the-Loop**. Taygeta ingiere documentos PDF/imagen, extrae preguntas y
figuras, transcribe matemáticas a Markdown/TeX, las clasifica curricularmente y
las almacena **sin soluciones ni respuestas correctas**. Después permite buscar
ejercicios reales (RAG) y generar exámenes de práctica compilados a PDF con
**Typst**.

> **Regla de oro:** Taygeta transcribe y estructura el contenido visible; nunca
> resuelve los ejercicios, no infiere respuestas correctas y no almacena claves
> de respuesta en ninguna capa (esquemas, staging, base de datos, RAG, PDFs ni
> salidas MCP).

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Requisitos](#requisitos)
4. [Instalación y configuración local](#instalación-y-configuración-local)
5. [Uso local](#uso-local)
6. [Despliegue con Docker Compose](#despliegue-con-docker-compose)
7. [Modelos OCR](#modelos-ocr)
8. [LLM remoto](#llm-remoto)
9. [Almacenamiento SeaweedFS (S3)](#almacenamiento-seaweedfs-s3)
10. [Endpoints HTTP y herramientas MCP](#endpoints-http-y-herramientas-mcp)
11. [Flujo Human-in-the-Loop](#flujo-human-in-the-loop)
12. [Respaldos y restauración](#respaldos-y-restauración)
13. [Verificación y pruebas](#verificación-y-pruebas)
14. [Limitaciones actuales](#limitaciones-actuales)
15. [Seguridad](#seguridad)

---

## Arquitectura

Ingesta **híbrida Edge-to-Cloud** en dos pases claramente separados:

```
 PDF/PNG/JPG
     │
     ▼
[1] Preparación  (ingest/image_extractor.py)
     PyMuPDF · rasterizado 300 DPI · detección de figuras · BBox 0-1000
     padding 1.5% · recortes temporales → subida a SeaweedFS
     │
     ▼
[2] Pase 1 · OCR local  (ingest/ocr_client.py)
     Servicio local OpenAI-compatible · transcribe a Markdown/TeX
     marca [ILEGIBLE] · NO clasifica ni resuelve
     │
     ▼
[3] Pase 2 · Estructuración remota  (ingest/pipeline.py)
     LLM remoto + instructor + Pydantic v2 → QuestionRaw
     │
     ▼
[4] Validación en dos capas  (ingest/validator.py)
     Capa 1 heurística local  +  Capa 2 semántica remota
     │
     ▼
[5] Staging (VALID / NEEDS_ATTENTION)  →  Revisión humana (Streamlit / chat MCP)
     │
     ▼
[6] Confirmación explícita → SQLite (taygeta.db, búsqueda FTS5)
     │
     ▼
[7] RAG + generación de exámenes (Typst)  →  PDF / ZIP de fuente
```

Componentes transversales:

- **Esquemas canónicos** (`core/schema.py`): `QuestionRaw`, `EnrichedQuestion`,
  `Choice`, `FigureReference`, `TranscriptionIntermediate`.
- **Transpilador** (`core/transpiler.py`): AST de Pandoc → Typst, con exportación
  LaTeX y compilación de validación con el CLI de Typst.
- **Persistencia** (`server/db.py`): SQLite + tabla virtual **FTS5** sincronizada
  por triggers; taxonomía y staging.
- **Objetos** (`server/object_storage.py`): cliente boto3 para SeaweedFS (S3).
- **UI** (`ui/app.py`): Streamlit HITL.
- **Servidor** (`server/mcp_server.py`): herramientas FastMCP, descargas HTTP y
  limpiador TTL.

---

## Estructura del repositorio

```
taygeta/
├── Dockerfile                     # imagen principal multi-stage (uv + Pandoc + Typst)
├── .dockerignore
├── docker-compose.yml             # base neutral de GPU
├── docker-compose.rocm.yml        # override AMD/ROCm
├── docker-compose.cuda.yml        # override NVIDIA/CUDA
├── docker/
│   ├── ocr/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.rocm
│   │   ├── Dockerfile.cuda
│   │   └── entrypoint.sh
│   └── seaweedfs/
│       └── s3.json.example        # credenciales S3 de ejemplo (sin secretos)
├── pyproject.toml
├── uv.lock
├── .env.example
└── src/taygeta/
    ├── core/     schema.py · settings.py · transpiler.py
    ├── ingest/   image_extractor.py · ocr_client.py · validator.py · pipeline.py
    ├── server/   db.py · object_storage.py · exam_generator.py · mcp_server.py
    └── ui/       app.py
```

---

## Requisitos

- **Python 3.11+** y [uv](https://docs.astral.sh/uv/).
- Para el flujo completo de compilación y OCR: **Pandoc** y **Typst** CLI
  (ya instalados y verificados en la imagen Docker).
- **Docker** y **Docker Compose v2** para el despliegue por contenedores.
- GPU opcional: AMD/ROCm o NVIDIA/CUDA para acelerar el OCR local.

---

## Instalación y configuración local

```bash
# 1) Dependencias (crea/actualiza uv.lock de forma coherente)
uv sync

# 2) Configuración: copia el ejemplo y completa los valores
cp .env.example .env
#   Edita .env (claves LLM, endpoint OCR, credenciales S3). NUNCA versiones .env.
```

Comprobar que el entorno quedó correcto:

```bash
uv run python -c "from taygeta.core.settings import get_settings; s=get_settings(); print(s.safe_summary())"
```

> `uv.lock` se versiona. Tras cambiar `pyproject.toml`, ejecuta `uv lock` o `uv sync`
> y confirma que `uv.lock` se actualiza de forma coherente.

Herramientas de calidad:

```bash
uv run ruff check src
uv run ruff format --check src
```

---

## Uso local

### Ingesta por CLI (`taygeta-ingest`)

```bash
uv run taygeta-ingest /ruta/a/examen.pdf
# con contexto correctivo:
uv run taygeta-ingest /ruta/a/examen.pdf --hint "La condición ilegible es g(x)=x+1"
```

Imprime un resumen JSON del staging (número de preguntas, estado y sesión).

### Servidor MCP + HTTP (`taygeta-server`)

```bash
uv run taygeta-server          # FastMCP con transporte streamable-http
# por defecto escucha en http://0.0.0.0:8000  (MCP_HOST / MCP_PORT)
```

Expone el protocolo MCP en `/mcp` y las rutas HTTP `/health`,
`/download/{filename}` y `/assets/{asset_id}`.

### Interfaz Streamlit (HITL)

```bash
uv run streamlit run src/taygeta/ui/app.py --server.port 8501 --server.address 0.0.0.0
# http://localhost:8501
```

---

## Despliegue con Docker Compose

Copia el ejemplo de entorno antes de levantar el stack:

```bash
cp .env.example .env      # completa LLM_*, OCR_MODEL_PATH, credenciales S3
```

### Base / CPU (neutral respecto de GPU)

```bash
docker compose up --build
```

Servicios: `mcp-server` (8000), `ui` (8501), `ocr` (CPU, solo red interna),
SeaweedFS (`seaweedfs-master`, `seaweedfs-volume`, `seaweedfs-filer`,
`seaweedfs-s3:8333`).

### Local con AMD/ROCm

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

### Producción con NVIDIA/CUDA

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

> **No mezcles** los overrides ROCm y CUDA en la misma ejecución.
> El archivo base es neutral de GPU: los dispositivos, el runtime NVIDIA y
> `HSA_OVERRIDE_GFX_VERSION` solo aparecen en los overrides.

Validar la configuración sin levantar los servicios:

```bash
docker compose config -q
docker compose -f docker-compose.yml -f docker-compose.rocm.yml config -q
docker compose -f docker-compose.yml -f docker-compose.cuda.yml config -q
```

Requisitos del host para GPU:

- **ROCm**: driver AMD + ROCm compatibles con la imagen base; acceso a `/dev/kfd`
  y `/dev/dri`; grupos `video`/`render` según el host. Fija `AMDGPU_TARGETS` al
  compilar la imagen OCR a tu arquitectura (p. ej. `gfx1100`).
- **CUDA**: driver NVIDIA + NVIDIA Container Toolkit. Verifica primero:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  Para reservar GPUs concretas, edita `docker-compose.cuda.yml` y cambia
  `count: all` por `count: 1` o `device_ids: ["0"]`.

---

## Modelos OCR

El servicio `ocr` usa `llama-server` (llama.cpp) con una API **OpenAI-compatible**
y soporte multimodal. **Los pesos no se incluyen en el repositorio ni en la
imagen**: viven en el volumen persistente `ocr-models`.

1. Descarga un modelo de visión adecuado para documentos científicos, por
   ejemplo **Qwen2.5-VL-7B-Instruct (GGUF)** con cuantización Q4/Q8, junto con su
   archivo **`mmproj`** (proyector visual, obligatorio para entrada multimodal).
2. Colócalos en el volumen `ocr-models` (por defecto en `/models`):

   ```bash
   # Ejemplo con huggingface-cli; ajusta repo/archivos a tu elección.
   docker run --rm -v taygeta_ocr-models:/models \
     ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
     bash -lc "pip install -q huggingface_hub && \
       huggingface-cli download <repo> <archivo>.gguf --local-dir /models && \
       huggingface-cli download <repo> <mmproj>.gguf --local-dir /models"
   ```

3. Ajusta en `.env`:

   ```dotenv
   OCR_MODEL=ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M
   OCR_MODEL_PATH=/models/<modelo>.gguf
   OCR_MMPROJ_PATH=/models/<mmproj>.gguf
   OCR_CONTEXT_SIZE=8192
   OCR_CONCURRENCY=1
   ```

   El nombre del volumen puede variar (`<proyecto>_ocr-models`); compruébalo con
   `docker volume ls`.

Si el contenedor arranca **sin pesos**, el entrypoint falla con un mensaje claro
en lugar de marcarse saludable.

**Verificar que el OCR ve imágenes y transcribe una fórmula:**

```bash
# 1) Salud del servicio
docker compose exec ocr curl -fsS http://localhost:8080/health

# 2) Salud + modelo desde el cliente de la aplicación
docker compose exec mcp-server python -c \
  "from taygeta.ingest.ocr_client import build_ocr_client; c=build_ocr_client(); print(c.health(), c.model)"

# 3) Transcripción real de una imagen de prueba
docker compose exec -T mcp-server python - <<'PY'
import io
from PIL import Image, ImageDraw
from taygeta.ingest.ocr_client import build_ocr_client

img = Image.new("RGB", (640, 140), "white")
ImageDraw.Draw(img).text((20, 50), "Ecuacion: x^2 + 2x + 1 = 0", fill="black")
buf = io.BytesIO(); img.save(buf, format="PNG")
result = build_ocr_client().transcribe_page(buf.getvalue(), page_index=0)
print(result.text)
PY
```

La respuesta debe ser Markdown/TeX con la fórmula transcrita (y `[ILEGIBLE]` si
algo no se pudo leer).

---

## LLM remoto

La estructuración (Pase 2) y la validación semántica (Capa 2) usan un LLM
**remoto** compatible con la API de OpenAI, configurado **independientemente** del
OCR:

```dotenv
LLM_API_KEY=            # SOLO la credencial (sin endpoint ni modelo)
LLM_BASE_URL=           # p.ej. https://mi-proveedor.example/v1
LLM_MODEL=              # identificador del modelo elegido por el administrador
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=2
```

Diagnóstico:

```bash
uv run python -c "from taygeta.core.settings import get_settings; s=get_settings(); print('configurado:', s.llm_configured); s.require_llm()"

# Comprobar el endpoint (lista de modelos) sin exponer la clave en logs:
curl -s "${LLM_BASE_URL%/}/models" -H "Authorization: Bearer ${LLM_API_KEY}" | head
```

> No se fija ningún proveedor ni modelo en el código. `LLM_API_KEY` es solo la
> credencial; no contiene el endpoint, el modelo ni parámetros.

---

## Almacenamiento SeaweedFS (S3)

Las figuras definitivas se guardan como objetos S3 con **clave estable**
(`s3://bucket/key`), nunca como rutas locales ni URLs firmadas persistentes.

```dotenv
S3_ENDPOINT_URL=http://seaweedfs-s3:8333   # endpoint interno (red Docker)
S3_BUCKET=taygeta-assets
S3_ACCESS_KEY_ID=change-me
S3_SECRET_ACCESS_KEY=change-me
S3_REGION=us-east-1
ASSET_PUBLIC_BASE_URL=                     # base pública/proxy opcional para la UI
ASSET_URL_TTL_SECONDS=300
```

- **Credenciales en dos sitios que deben coincidir**: `docker/seaweedfs/s3.json.example`
  (identidades del gateway S3) y `.env` (`S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`).
  Cámbialas antes de cualquier despliegue no local.
- **Endpoint interno vs público**: los contenedores usan `http://seaweedfs-s3:8333`.
  El navegador nunca debe apuntar a un host interno de Docker; para mostrar figuras
  usa `ASSET_PUBLIC_BASE_URL` o el endpoint `/assets/{asset_id}` de la aplicación.
  No se reescribe el host de una URL firmada después de generarla (invalidaría la firma).

**Verificar el bucket (crear/leer/borrar un objeto):**

```bash
docker compose exec mcp-server python - <<'PY'
from taygeta.server.object_storage import ObjectStorage
s = ObjectStorage()
s.ensure_bucket()
uri = s.store_asset_bytes(b"prueba", prefix="checks", extension=".txt")
print("subido:", uri, "existe:", bool(uri))
key = uri.split("/", 3)[3]
print("leido:", s.get_bytes(key))
s.delete(key); print("borrado:", not s.exists(key))
PY
```

---

## Endpoints HTTP y herramientas MCP

Rutas HTTP en el puerto 8000 (`mcp-server`):

| Ruta                         | Descripción                                              |
|------------------------------|----------------------------------------------------------|
| `GET /health`                | Estado para healthchecks.                                |
| `POST /mcp`                  | Transporte MCP (streamable-http).                        |
| `GET /download/{filename}`   | Sirve PDFs/ZIP temporales de `/app/downloads` (valida).  |
| `GET /assets/{asset_id}`     | Devuelve una figura desde SeaweedFS (id validado).       |

Herramientas FastMCP:

| Herramienta | Descripción |
|---|---|
| `stage_exam_upload(file_path)` | Procesa un archivo de `uploads/` y crea una sesión de staging. |
| `refine_staged_question(session_id, q_index, user_hint, manual_overrides)` | Reanaliza con contexto correctivo; conserva overrides. |
| `confirm_staged_upload(session_id)` | Persiste las preguntas aprobadas (idempotente). |
| `search_questions(course, topic, difficulty, limit)` | Recuperación RAG por curso/tema/dificultad. |
| `list_taxonomy()` | Catálogo de cursos, temas, subtemas y prerrequisitos. |
| `generate_practice_exam(course, topics, difficulty, num_questions)` | Genera y compila un examen en PDF (Typst). |
| `get_exam_source(exam_id, format)` | ZIP con `main.typ`/`main.tex`, `assets/` y `README.txt`. |

Probar la descarga tras generar un examen:

```bash
EXAM_ID=$(docker compose exec -T mcp-server ... )   # devuelve el exam_id
curl -fL "http://localhost:8000/download/exams/${EXAM_ID}.pdf" -o examen.pdf
curl -fL "http://localhost:8000/download/fuente_examen_${EXAM_ID}.zip" -o fuente.zip
```

---

## Flujo Human-in-the-Loop

1. **Sube** un PDF/PNG/JPG (UI o `stage_exam_upload`). Se guarda en `uploads/`.
2. El pipeline ejecuta el doble pase y la validación en dos capas; cada pregunta
   queda en staging como `VALID` o `NEEDS_ATTENTION` (con motivo técnico).
3. **Revisa** en la UI: enunciado Markdown renderizado en paralelo con la imagen
   original (recorte de figura desde SeaweedFS o página rasterizada).
4. **Edita** enunciado, tipo, alternativas, dificultad (1–5), curso/tema/subtema y
   prerrequisitos. Aporta contexto correctivo y pulsa **Reanalizar** si hace falta.
5. **Confirma** para persistir en `taygeta.db`. La corrección manual prevalece
   sobre la transcripción automática.

> Ninguna pregunta se guarda en la base definitiva sin confirmación explícita.
> Un fallo de OCR o de validación **nunca** se oculta marcando la pregunta como válida.

---

## Respaldos y restauración

### SQLite (`taygeta-data`)

Copia consistente con la API de backup de SQLite (evita copiar con escrituras en vuelo):

```bash
docker compose exec -T mcp-server python - <<'PY'
import sqlite3
src = sqlite3.connect("/app/data/taygeta.db")
dst = sqlite3.connect("/app/data/taygeta-backup.db")
with dst:
    src.backup(dst)
src.close(); dst.close()
print("backup creado en /app/data/taygeta-backup.db")
PY
docker compose cp mcp-server:/app/data/taygeta-backup.db ./taygeta-backup.db
```

Restaurar (con el stack parado para evitar escrituras):

```bash
docker compose stop mcp-server ui
docker compose cp ./taygeta-backup.db mcp-server:/app/data/taygeta.db
docker compose start mcp-server ui
```

Alternativa por volumen (snapshot en frío):

```bash
docker compose stop mcp-server ui
docker run --rm -v "$(basename "$PWD")_taygeta-data":/data -v "$PWD":/backup alpine \
  tar czf /backup/taygeta-data.tgz -C /data .
docker compose start mcp-server ui
```

### SeaweedFS

Respalda los volúmenes persistentes de master, volume y filer (en frío):

```bash
docker compose stop seaweedfs-s3 seaweedfs-filer seaweedfs-volume seaweedfs-master
for v in seaweedfs-master-data seaweedfs-volume-data seaweedfs-filer-data; do
  docker run --rm -v "$(basename "$PWD")_${v}":/data -v "$PWD":/backup alpine \
    tar czf "/backup/${v}.tgz" -C /data .
done
docker compose start seaweedfs-master seaweedfs-volume seaweedfs-filer seaweedfs-s3
```

Restaurar extrayendo cada `.tgz` en su volumen correspondiente (con los servicios
parados). El bucket y las credenciales se reconfiguran vía `docker/seaweedfs/s3.json`.

---

## Verificación y pruebas

Comprobaciones estáticas:

```bash
uv run ruff check src
uv run ruff format --check src
```

Verificaciones de integración realizadas durante el desarrollo (contra
implementaciones reales de SQLite, PyMuPDF, Pandoc y Typst; con dobles solo para
los servicios externos no desplegados):

- **Esquemas**: validación Pydantic v2, slugs `^[a-z0-9_]+$`, dificultad 1–5,
  correspondencia placeholder/figura y ausencia de campos de solución.
- **SQLite/FTS5**: inicialización idempotente, sincronización de `questions_fts`
  en altas/actualizaciones/borrados y consultas parametrizadas.
- **Extracción**: rasterizado 300 DPI, desnormalización con padding 1.5 %, recorte
  de figuras.
- **Transpilador**: conversión TeX→Typst de matrices `pmatrix`, `cases`, fracciones,
  raíces, scripts y sumatorias, y **compilación real** de PDF con Typst.
- **Pipeline/validación**: clasificación estricta `VALID`/`NEEDS_ATTENTION` y
  ausencia de respuestas correctas persistidas.
- **MCP/exámenes**: generación de PDF, ZIP de fuente (Typst y LaTeX) y rutas de
  descarga; `docker compose config` para base + ROCm + CUDA.

> La suite automatizada `tests/` (pytest) con estos casos, exigida por la
> especificación, **está pendiente** de incorporar al repositorio. Hasta entonces,
> estas comprobaciones se ejecutan como scripts de verificación en el entorno de
> desarrollo. Los casos que dependen de GPU o de un proveedor LLM/OCR reales solo
> pueden completarse en el host de despliegue.

---

## Limitaciones actuales

- **Transcripción OCR**: depende del modelo de visión local. Texto o símbolos
  ilegibles se marcan como `[ILEGIBLE]` y requieren revisión humana; no se inventa
  contenido.
- **Conversión TeX → Typst**: cubre los constructos acordados (delimitadores
  inline/display, `cases`, `pmatrix`/entornos matriciales, fracciones, raíces,
  scripts, sumatorias/integrales/límites, `aligned` y `substack`). **No** garantiza
  la traducción de macros TeX arbitrarias: ante una construcción no soportada el
  transpilador falla de forma controlada, emite un error descriptivo y conserva el
  original para su recuperación o exportación LaTeX.
- **Espaciado matemático**: TeX trata cada letra como un símbolo; Typst une letras
  adyacentes en un identificador. El transpilador inserta separadores estructurales
  para preservar la semántica, pero casos muy específicos de espaciado pueden
  diferir visualmente.
- **Alineado (`aligned`)**: se aproxima concatenando celdas por fila; es una
  conversión best-effort dentro del alcance declarado.
- **Servicios externos**: la validación semántica y la estructuración requieren un
  LLM remoto; si no está disponible, las preguntas quedan en `NEEDS_ATTENTION` con
  el motivo técnico (no se pierden).
- **Verificación de GPU/modelos**: la detección real de GPU y la transcripción con
  el modelo descargado deben comprobarse en el host de despliegue.

---

## Seguridad

- **Aislamiento y persistencia**: la base de datos vive en `/app/data`, los
  documentos en `/app/uploads` y los artefactos temporales en `/app/downloads`;
  las figuras definitivas viven en SeaweedFS. Nada de esto se versiona en Git.
- **S3**: el gateway usa credenciales configuradas y no permite lectura pública
  anónima. No se persisten URLs firmadas; se generan bajo demanda.
- **Rutas**: las entradas de MCP (archivos de entrada) y las descargas validan que
  la ruta permanezca dentro del directorio permitido (anti-traversal).
- **Procesos**: se invocan `pandoc` y `typst` con argumentos en lista, nunca con
  `shell=True`.
- **Secretos**: `LLM_API_KEY`, credenciales S3 y tokens nunca se imprimen en logs;
  usa `Settings.safe_summary()` (enmascara secretos) para diagnósticos.
- **Sin soluciones**: no se almacenan respuestas correctas en la base de datos,
  staging, RAG, PDFs generados ni en las salidas MCP.
- **Limpieza**: el limpiador de `/app/downloads` respeta un TTL de 60 minutos y no
  toca la base de datos, las cargas originales ni los objetos del bucket.

---

## Licencia

Uso propietario (`license = "Proprietary"` en `pyproject.toml`).
