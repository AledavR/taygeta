# syntax=docker/dockerfile:1
#
# Imagen principal de la aplicación Taygeta (sección 11.2).
# Multi-stage con uv: se construye un entorno virtual reproducible y se copia a
# una imagen de runtime con Pandoc y Typst CLI instalados y verificados.
# No incluye pesos de modelos, base de datos, documentos ni figuras.

# ---------------------------------------------------------------------------
# Etapa 1: resolución e instalación de dependencias con uv
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Capa de dependencias cacheada respecto a uv.lock / pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Código y configuración no secreta.
COPY pyproject.toml uv.lock ./
COPY src ./src

# Instala el proyecto dentro del entorno virtual.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Etapa 2: imagen de runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ARG TYPST_VERSION=0.15.1
ARG PANDOC_VERSION=3.12.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# Dependencias del sistema: curl (healthchecks/descargas), ca-certificates y xz.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/*

# CLI de Pandoc (paquete oficial .deb del release indicado).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    pkg="pandoc-${PANDOC_VERSION}-1-${arch}.deb"; \
    curl -fsSL -o /tmp/pandoc.deb \
      "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/${pkg}"; \
    dpkg -i /tmp/pandoc.deb; \
    rm -f /tmp/pandoc.deb; \
    pandoc --version

# CLI de Typst (binario estático oficial musl).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) target="x86_64-unknown-linux-musl" ;; \
      arm64) target="aarch64-unknown-linux-musl" ;; \
      *) echo "arquitectura no soportada: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/typst.tar.xz \
      "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-${target}.tar.xz"; \
    tar -xJf /tmp/typst.tar.xz -C /tmp; \
    install -m 0755 "/tmp/typst-${target}/typst" /usr/local/bin/typst; \
    rm -rf /tmp/typst.tar.xz "/tmp/typst-${target}"; \
    typst --version

# Usuario no root.
RUN groupadd --system --gid 1001 taygeta \
 && useradd --system --uid 1001 --gid taygeta --create-home taygeta

WORKDIR /app

# Entorno virtual ya resuelto + código.
COPY --from=builder --chown=taygeta:taygeta /app/.venv /app/.venv
COPY --chown=taygeta:taygeta pyproject.toml uv.lock ./
COPY --chown=taygeta:taygeta src ./src

# Directorios de runtime (se sobreescriben con volúmenes persistentes).
RUN mkdir -p /app/data /app/uploads /app/downloads \
 && chown -R taygeta:taygeta /app

USER taygeta

# 8000: MCP/HTTP de descargas. 8501: Streamlit (servicio `ui`, misma imagen).
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Comando de inicio del servidor MCP (script instalado por el proyecto).
# La UI se inicia con: streamlit run src/taygeta/ui/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["taygeta-server"]
