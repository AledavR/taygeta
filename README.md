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
[3] Estructuración en un solo pase  (agents/structurer.py)
     LLM remoto + instructor + Pydantic v2 → EnrichedQuestion
     (TeX formateado + course/topic/subtopic + difficulty + prerequisites)
     │
     ▼
[4] Validación en dos capas  (ingest/validator.py)
     Capa 1 heurística local  +  Capa 2 semántica remota
     │
     ▼
[5] Staging (VALID / NEEDS_ATTENTION)  →  Revisión humana (UI NiceGUI / chat MCP)
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
- **Transpilador** (`core/transpiler.py`): AST de Pandoc → Typst; las fórmulas
  TeX se delegan a **tylax** (conversor LaTeX↔Typst en Rust). Exportación LaTeX
  y compilación de validación con el CLI de Typst.
- **Persistencia** (`server/db.py`): SQLite + tabla virtual **FTS5** sincronizada
  por triggers; taxonomía y staging.
- **Objetos** (`server/object_storage.py`): cliente boto3 para SeaweedFS (S3).
- **UI** (`ui/nice/`): revisión Human-in-the-Loop en **NiceGUI** (FastAPI).
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
    ├── core/     schema.py · settings.py · urls.py · transpiler.py
    ├── agents/   structurer.py
    ├── ingest/   image_extractor.py · ocr_client.py · validator.py · pipeline.py
    ├── server/   db.py · object_storage.py · exam_generator.py · mcp_server.py
    └── ui/       nice/ (app.py · question_panel.py · figures.py · state.py) · shared.py · figure_editing.py
```

---

## Requisitos

- **Python 3.11+** y [uv](https://docs.astral.sh/uv/).
- Para el flujo completo de compilación y OCR: **Pandoc** y **Typst** CLI
  (ya instalados y verificados en la imagen Docker).
- Conversor TeX→Typst **tylax**: se instala automáticamente como dependencia de
  Python (wheel nativa de Rust, `abi3`).
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

### Interfaz de revisión Human-in-the-Loop (NiceGUI)

```bash
uv run taygeta-ui            # o: uv run python -m taygeta.ui.nice
# http://localhost:8501
```

La UI permite: subir un PDF/PNG/JPG, procesarlo con el pipeline de doble pase
en background (sin bloquear la interfaz), revisar cada pregunta (editor
Markdown/TeX con previsualización, taxonomía con comboboxes de curso/tema,
opciones avanzadas con reanálisis in-place) y gestionar las figuras a la
derecha (vistas «Figuras» / «Página original», recorte por arrastre, alta y
borrado) antes de confirmar explícitamente la persistencia en SQLite.

Host y puerto se ajustan con `TAYGETA_UI_HOST` / `TAYGETA_UI_PORT`.

---

## Despliegue con Docker Compose

Copia el ejemplo de entorno antes de levantar el stack:

```bash
cp .env.example .env      # completa LLM_*, OCR_MODEL_PATH, credenciales S3
```

Configuración dual (local/producción):

```dotenv
APP_ENV=local                       # 'local' o 'production'
DATABASE_URL=                       # opcional; si está vacía se usa DB_PATH
                                    # p. ej. sqlite:////app/data/taygeta.db
```

- `APP_ENV` marca el entorno de ejecución (`Settings.is_production`).
- `DATABASE_URL` permite apuntar a otra ruta SQLite sin tocar `DB_PATH`; las URLs
  no SQLite no están soportadas por la capa actual y producen un error explícito.

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
  y `/dev/dri`; grupos `video`/`render` según el host. Ajusta `AMDGPU_TARGETS`
  (p. ej. `gfx1100` para RX 7900) antes de compilar la imagen OCR. Solo si ROCm
  no detecta la GPU de forma nativa define `HSA_OVERRIDE_GFX_VERSION` (p. ej.
  `10.3.0` para RDNA2); **nunca lo dejes vacío**: un valor vacío desactiva la
  detección y el OCR cae a CPU. Con `OCR_REQUIRE_GPU=1` (ya activo en el
  override) el contenedor **aborta con un mensaje claro** si no ve la GPU, en
  lugar de arrancar degradado.
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

**Descarga automática desde Hugging Face**: si los pesos no están presentes, el
entrypoint los descarga al arrancar usando `OCR_MODEL_URL` y `OCR_MMPROJ_URL`. Por
defecto se usa **Qwen2-VL-7B-Instruct (GGUF, Q4_K_M)** más su **`mmproj`**:

```dotenv
OCR_MODEL=Qwen2-VL-7B-Instruct-Q4_K_M
OCR_MODEL_PATH=/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf
OCR_MODEL_URL=https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf
OCR_MMPROJ_PATH=/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf
OCR_MMPROJ_URL=https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf
OCR_CONTEXT_SIZE=8192
OCR_CONCURRENCY=1
```

- Las descargas van a un fichero `.part` y se renombran al completar (reanudables):
  una interrupción no deja un fichero a medias.
- Si el fichero **ya existe** en el volumen, no se vuelve a descargar.
- También puedes **montar los pesos manualmente** en `ocr-models` (p. ej. con
  `huggingface-cli`) y ajustar las rutas/URLs en `.env`.
- Si no hay pesos y no se definen URLs, el entrypoint falla con un mensaje claro
  en lugar de marcarse saludable.
- El primer arranque descarga varios GB (modelo + proyector); el nombre del volumen
  puede variar (`<proyecto>_ocr-models`), compruébalo con `docker volume ls`.

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

La estructuración en un solo pase (que también infiere ``course``, ``topic``,
``subtopic``, ``difficulty`` y ``prerequisites``) y la validación semántica
(Capa 2) usan un LLM **remoto** compatible con la API de OpenAI, configurado
**independientemente** del OCR:

```dotenv
LLM_API_KEY=            # SOLO la credencial (sin endpoint ni modelo)
LLM_BASE_URL=           # p.ej. https://mi-proveedor.example/v1
LLM_MODEL=              # identificador del modelo elegido por el administrador
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=2
LLM_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   # embeddings locales
DUPLICATE_SIMILARITY_THRESHOLD=0.90          # umbral coseno (0-1) de candidato a duplicado
```

Diagnóstico:

```bash
uv run python -c "from taygeta.core.settings import get_settings; s=get_settings(); print('configurado:', s.llm_configured); s.require_llm()"

# Comprobar el endpoint (lista de modelos) sin exponer la clave en logs:
curl -s "${LLM_BASE_URL%/}/models" -H "Authorization: Bearer ${LLM_API_KEY}" | head
```

> No se fija ningún proveedor ni modelo en el código. `LLM_API_KEY` es solo la
> credencial; no contiene el endpoint, el modelo ni parámetros.

### Deduplicación semántica

Además de la deduplicación por hash exacto, la ingesta calcula un **embedding**
**local** del enunciado y sus alternativas con `sentence-transformers`
(`LLM_EMBEDDING_MODEL`, sin API externa ni `LLM_API_KEY`) y lo compara por
similitud coseno contra las preguntas ya almacenadas del mismo curso. Si supera
`DUPLICATE_SIMILARITY_THRESHOLD`, la pregunta queda marcada como candidata y un
juez LLM decide si es matemáticamente la misma (ignorando variaciones de
notación/OCR). Si se confirma, se descarta el texto nuevo y se fusionan los
`source_reference` y prerrequisitos en la pregunta existente. Nunca se
almacenan soluciones ni respuestas correctas.

Los pesos del modelo se descargan/cachean la primera vez que se generan
embeddings (igual que los del OCR); también puede usarse una ruta local en
`LLM_EMBEDDING_MODEL`.

Para migrar preguntas ya existentes (generar embeddings y fusionar duplicados):

```bash
uv run python scripts/deduplicate_questions.py --db /ruta/a/taygeta.db --dry-run
uv run python scripts/deduplicate_questions.py --db /ruta/a/taygeta.db
```

Si el modelo no está disponible (sin pesos ni red), la deduplicación semántica
se desactiva de forma controlada y se conserva la de hash exacto.

---

## Almacenamiento SeaweedFS (S3)

Las figuras definitivas se guardan como objetos S3 con **ruta relativa estable**
(`/buckets/{bucket}/{key}`, p. ej. `/buckets/exams/img.png`), nunca como rutas
locales ni URLs firmadas persistentes. La ruta relativa es portable entre entornos
y se resuelve a URL completa solo al enviar la información al frontend o al MCP.

```dotenv
S3_ENDPOINT_URL=http://seaweedfs-s3:8333   # endpoint interno (red Docker)
S3_PUBLIC_ENDPOINT=http://localhost:8333   # endpoint público para el navegador (UI)
S3_PUBLIC_BASE_URL=                        # base pública para /buckets/{bucket}/{key}
S3_BUCKET=taygeta-assets
S3_ACCESS_KEY_ID=change-me
S3_SECRET_ACCESS_KEY=change-me
S3_REGION=us-east-1
ASSET_PUBLIC_BASE_URL=                     # base pública/proxy opcional (legacy, UI)
ASSET_URL_TTL_SECONDS=300
```

- **Credenciales de un único origen**: el gateway S3 genera su configuración IAM a
  partir de `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` del `.env`, de modo que las
  credenciales de la aplicación y del gateway siempre coinciden. El esquema del
  fichero IAM queda documentado en `docker/seaweedfs/s3.json.example`.
  Cámbialas antes de cualquier despliegue no local.
- **Rutas relativas → URLs públicas**: los modelos almacenan
  `/buckets/{bucket}/{key}`. Para enviar la información al frontend o al cliente
  MCP se compone la URL completa concatenando `S3_PUBLIC_BASE_URL`
  (`build_public_url` en `core/urls.py`); en la UI también puedes usar
  `ASSET_PUBLIC_BASE_URL`, el endpoint `/assets/{asset_id}` de la aplicación o
  `S3_PUBLIC_ENDPOINT` (reescritura del host interno). No se reescribe el host de
  una URL firmada después de generarla (invalidaría la firma).

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
3. **Revisa** en la UI: la columna izquierda muestra el enunciado Markdown con su
   previsualización renderizada, el tipo/alternativas y los metadatos; la
   columna derecha se dedica a las figuras con dos pestañas, **Figuras** y
   **Página original** (la página del documento fuente, útil para detectar
   figuras omitidas). Cada figura puede **eliminarse** (se limpia su `{{fig:N}}`
   del enunciado) o **recortarse manualmente** dibujando un bounding box sobre
   la página original; el recorte se sobrescribe en SeaweedFS y la vista previa
   se refresca al instante. Si el modelo omitió una figura real (falso
   negativo), **➕ Añadir nueva figura** carga la página completa, permite
   dibujar el recuadro sobre el gráfico faltante y lo guarda como el siguiente
   `fig:N` libre, insertando `{{fig:N}}` al final del enunciado para que lo
   muevas a su lugar.
4. **Edita** enunciado, tipo, alternativas, dificultad (1–5) y prerrequisitos.
   Curso y Tema son comboboxes: selecciona un slug del catálogo o usa
   **✏️ Escribir manualmente...** para crear uno nuevo. Aporta contexto
   correctivo y pulsa **Reanalizar** si hace falta (todo en el panel de la
   izquierda, con **Opciones Avanzadas** para el contexto y los prerrequisitos
   nuevos).
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
  de figuras, descarte de candidatos que cubren la página completa (escaneos,
  fondos o dibujos de marco) y re-recorte manual desde la UI.
- **Transpilador**: conversión TeX→Typst delegada a **tylax** (corpus de oro de
  fórmulas: fracciones, raíces, matrices `pmatrix`, `cases`, `aligned`, scripts,
  sumatorias/integrales/límites) y **compilación real** de PDF con Typst.
- **Pipeline/validación**: clasificación estricta `VALID`/`NEEDS_ATTENTION` y
  ausencia de respuestas correctas persistidas.
- **MCP/exámenes**: generación de PDF, ZIP de fuente (Typst y LaTeX) y rutas de
  descarga; `docker compose config` para base + ROCm + CUDA.

> La suite automatizada `tests/` (pytest) empieza a incorporar estos casos
> (sanitización de delimitadores, conversión TeX→Typst con tylax y edición de
> figuras). Los casos que dependen de GPU o de un proveedor LLM/OCR reales solo
> pueden completarse en el host de despliegue.

---

## Limitaciones actuales

- **Transcripción OCR**: depende del modelo de visión local. Texto o símbolos
  ilegibles se marcan como `[ILEGIBLE]` y requieren revisión humana; no se inventa
  contenido.
- **Conversión TeX → Typst (tylax)**: cubre un amplio catálogo de símbolos y
  macros LaTeX (~700 símbolos, `\newcommand`/`\def`, matrices, `cases`,
  fracciones, raíces, scripts, sumatorias/integrales/límites). tylax es **no
  estricto**: un comando desconocido se conserva como texto crudo (no lanza) y se
  detecta en la compilación real con Typst (`unknown variable: ...`); ante un
  LaTeX no analizable, tylax emite un marcador `/* LaTeX Error ... */` que el
  transpilador eleva a error descriptivo conservando el original.
- **Espaciado matemático**: tylax separa los átomos de forma estructural, pero
  casos muy específicos de espaciado pueden diferir visualmente (p. ej.
  `\text{...}` pierde los espacios de relleno de los extremos).
- **Alineado (`aligned`)**: tylax emite el alineado nativo de Typst (`&`/`\`);
  conviene revisar visualmente los casos de más de una columna.
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
  `shell=True`; la conversión TeX→Typst corre **en proceso** con `tylax`
  (librería Rust nativa, sin subprocesos adicionales).
- **Secretos**: `LLM_API_KEY`, credenciales S3 y tokens nunca se imprimen en logs;
  usa `Settings.safe_summary()` (enmascara secretos) para diagnósticos.
- **Sin soluciones**: no se almacenan respuestas correctas en la base de datos,
  staging, RAG, PDFs generados ni en las salidas MCP.
- **Limpieza**: el limpiador de `/app/downloads` respeta un TTL de 60 minutos y no
  toca la base de datos, las cargas originales ni los objetos del bucket.

---

## Licencia

Uso propietario (`license = "Proprietary"` en `pyproject.toml`).
