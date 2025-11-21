"""Simple migration entrypoint for production/Cloud Run.

This script reuses core.db.ensure_schema() to create/upgrade the
required database tables. It is safe to run multiple times.

Usage (local):

    python -m scripts.migrate

In Cloud Run, you can run this as a one-off job using the same image
that powers the backend service, provided DATABASE_URL points at the
Cloud SQL Postgres instance.
"""
from __future__ import annotations

import logging

from core.db import ensure_schema


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("running ensure_schema migration")
    ensure_schema()
    logging.info("migration complete")


if __name__ == "__main__":
    main()
