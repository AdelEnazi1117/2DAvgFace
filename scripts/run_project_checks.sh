#!/usr/bin/env bash
set -euo pipefail

SKIP_TESTS=false
WITH_BUILD=false

for arg in "$@"; do
  case "$arg" in
    --skip-tests)
      SKIP_TESTS=true
      ;;
    --with-build)
      WITH_BUILD=true
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--skip-tests] [--with-build]"
      exit 1
      ;;
  esac
done

echo "==> Verifying environment"
python scripts/verify_setup.py

echo "==> Compiling Python sources"
python -m compileall src apps >/dev/null

if [ "$SKIP_TESTS" = false ]; then
  if [ -d tests ]; then
    if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)
PY
    then
      echo "==> Running tests"
      pytest -q
    else
      echo "==> pytest not installed; skipping tests"
    fi
  else
    echo "==> No tests/ directory found; skipping tests"
  fi
else
  echo "==> Tests skipped"
fi

if [ "$WITH_BUILD" = true ]; then
  if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("build") else 1)
PY
  then
    echo "==> Building package"
    python -m build
  else
    echo "==> python -m build not available; skipping build"
  fi
fi
