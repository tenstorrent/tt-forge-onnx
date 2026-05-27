# Documentation

## Prerequisites

- Docs use `sphinx_rtd_theme` with the canonical Tenstorrent UI from [docs.tenstorrent.com](https://docs.tenstorrent.com/_static/tt_theme.css) (via `shared/sphinx/tt_theme.py` in the monorepo). Vendored assets under `docs/sphinx/_static/` support offline builds (`TT_DOCS_THEME_CSS=local`).
- User guides live in `docs/src/` (Markdown). For Sphinx HTML preview, `docs/sphinx/src` is a symlink to `docs/src/`.
- CMake’s `docs` target uses [mdBook](https://github.com/rust-lang/mdBook) for the published book build.
- Manual install: `python -m pip install -r docs/requirements.txt`.

## Build Sphinx HTML locally

Use a project venv so system `sphinx-build` is not picked up:

```bash
cd /path/to/tt-forge-onnx
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt

cd docs/sphinx
TT_DOCS_THEME_CSS=local python -m sphinx -b html . _build/html
python -m http.server 8000 -d _build/html
```

Open http://localhost:8000/index.html — the sidebar follows `docs/src/SUMMARY.md` (e.g. **Getting Started** → Docker / build from source, **Testing** → Pytest / perf).

If `docs/sphinx/src` is missing, recreate the symlink: `ln -sf ../src docs/sphinx/src`.

## Build (CMake / mdBook)

```bash
cmake --build build --target docs
mdbook serve build/docs
```

## Theme assets

- Shared theme helper: monorepo `shared/sphinx/tt_theme.py`
- Project overrides: `docs/sphinx/_static/`, `docs/sphinx/_templates/`
