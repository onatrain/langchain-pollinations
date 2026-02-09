# Tooling (uv + Hatchling + src layout)

## Requisitos

- Tener `uv` instalado.
- Tener Python 3.11+ disponible.

## Estructura esperada del proyecto

Asegurarse que el paquete importable exista dentro de `src/`:

```text
langchain-pollinations/
  pyproject.toml
  src/
    langchain_pollinations/
      __init__.py
      py.typed
      chat.py
      image.py
      account.py
      models.py
  tests/
    unit/
    integration/
```

## `pyproject.toml` mínimo (src layout + Hatchling)

Verificar que `pyproject.toml` contenga, como mínimo, lo siguiente:

```toml
[build-system]
requires = ["hatchling>=1.24.0"]
build-backend = "hatchling.build"

[project]
name = "langchain-pollinations"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langchain-core>=1.0.0",
  "pydantic>=2.6.0",
  "httpx>=0.27.0",
  "typing-extensions>=4.10.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-asyncio>=0.23.0",
  "respx>=0.21.0",
  "python-dotenv>=1.0.0",
  "mypy>=1.8.0",
  "ruff>=0.4.0",
  "isort>=5.13.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/langchain_pollinations"]

[tool.hatch.build]
include = [
  "src/langchain_pollinations/py.typed",
]

[project.entry-points."langchain.chatmodels"]
pollinations = "langchain_pollinations.chat:ChatPollinations"

[tool.pytest.ini_options]
markers = [
  "integration: tests que requieren POLLINATIONS_API_KEY y red",
]
asyncio_mode = "auto"
```

## Setup de ambiente (desarrollo)

Desde la raíz del proyecto (donde está `pyproject.toml`):

1) Crear el entorno virtual (recomendado para flujo estable):

```bash
uv venv
```

2) Instalar el proyecto en modo editable (necesario para que `uv run python ...` encuentre `src/langchain_pollinations`):

```bash
uv pip install -e ".[dev]"
```

3) Validar el import:

```bash
uv run python -c "from langchain_pollinations import ChatPollinations; print('OK')"
```

## Workflow con lock/sync (opcional)

Si se desea mantener `uv.lock`:

1) Generar/actualizar lock:

```bash
uv lock
```

2) Sincronizar dependencias (incluyendo extras de desarrollo):

```bash
uv sync --all-extras
```

3) Asegurar editable del proyecto (si aplica a tu flujo):

```bash
uv pip install -e ".[dev]"
```

## Comandos comunes

- Ruff:

```bash
uv run ruff check .
```

- Mypy:

```bash
uv run mypy src
```

- Unit tests:

```bash
uv run pytest -q tests/unit
```

- Integration tests (requiere `POLLINATIONS_API_KEY` en el entorno o `.env`):

```bash
uv run pytest -q -m integration tests/integration
```

## Troubleshooting: `ModuleNotFoundError: No module named 'langchain_pollinations'`

Ejecutar estos checks desde la raíz del repo:

1) Confirmar que el paquete existe:

```bash
ls -la src/langchain_pollinations/__init__.py
```

2) Confirmar instalación editable:

```bash
uv pip show langchain-pollinations
```

3) Reinstalar editable:

```bash
uv pip uninstall -y langchain-pollinations
uv pip install -e ".[dev]"
```

4) Fallback temporal sin instalación (solo para diagnosticar):

```bash
uv run env PYTHONPATH=src python -c "from langchain_pollinations import ChatPollinations; print('OK')"
```