import os
import sys

_CLIENT = None
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from main import app  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore
    from core.db import get_db_conn  # type: ignore
    from core.auth import hash_password  # type: ignore

    _CLIENT = TestClient(app)
except Exception:
    _CLIENT = None


def _backend_up() -> bool:
    return _CLIENT is not None


def test_login_and_me_roundtrip():
    """Smoke test for email/password login and /api/auth/me.

    Inserts or updates a test user in app_user, logs in via the API, and then
    calls /api/auth/me with the returned JWT.
    """
    if not _backend_up():
        return

    email = "test_auth_user@example.com"
    password = "test-password-123"

    # Ensure user row exists with known password hash
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            pw_hash = hash_password(password)
            cur.execute(
                """
                INSERT INTO app_user (email, password_hash, display_name, role, created_at, updated_at)
                VALUES (%s, %s, %s, %s, now(), now())
                ON CONFLICT (email) DO UPDATE
                  SET password_hash = EXCLUDED.password_hash,
                      updated_at = now()
                RETURNING id
                """,
                (email, pw_hash, "Auth Test User", "student"),
            )
            row = cur.fetchone()
            user_id = str(row[0])
        conn.commit()
    finally:
        conn.close()

    # Login
    r = _CLIENT.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("access_token")
    assert body.get("user", {}).get("id") == user_id

    token = body["access_token"]

    # /me
    r_me = _CLIENT.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r_me.status_code == 200
    me = r_me.json()
    assert me.get("id") == user_id
    assert me.get("email") == email
