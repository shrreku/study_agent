from __future__ import annotations

import os
import sys
from typing import Any
from pathlib import Path


def get_db_conn() -> Any:
    """Return a psycopg2 connection using the backend/core db helper.

    This mirrors the path-detection logic used by other scripts so the
    same code works both in local development (with a top-level
    `backend/` directory) and inside the Docker image (where the app is
    usually mounted at `/app`).
    """
    root = Path(__file__).resolve().parents[3]
    backend_dir = root / "backend"

    # In Docker, backend modules may live directly under /app
    app_dir = Path("/app")

    candidates = []
    if backend_dir.exists():
        candidates.append(str(backend_dir))
    if app_dir.exists():
        candidates.append(str(app_dir))

    for path in candidates:
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        # Try local-style import first
        from backend.core.db import get_db_conn as _get_db_conn  # type: ignore
    except Exception:
        # Fall back to app-style import (inside container)
        from core.db import get_db_conn as _get_db_conn  # type: ignore

    return _get_db_conn()
