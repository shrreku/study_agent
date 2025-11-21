from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from psycopg2.extras import RealDictCursor

from core.db import get_db_conn
from core.auth import hash_password, verify_password, create_access_token, require_auth


router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _build_user_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "display_name": row.get("display_name"),
        "role": row.get("role") or "student",
    }


@router.post("/api/auth/register")
async def register(body: RegisterRequest):
    if len(body.password or "") < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM app_user WHERE email=%s", (body.email,))
            existing = cur.fetchone()
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")

            pw_hash = hash_password(body.password)
            cur.execute(
                """
                INSERT INTO app_user (email, password_hash, display_name, role, created_at, updated_at)
                VALUES (%s, %s, %s, %s, now(), now())
                RETURNING id, email, display_name, role
                """,
                (body.email, pw_hash, body.display_name, "student"),
            )
            row = cur.fetchone()
            conn.commit()
    finally:
        conn.close()

    user = _build_user_payload(row)
    token = create_access_token(user_id=user["id"], email=user["email"])
    return {"user": user, "access_token": token, "token_type": "bearer"}


@router.post("/api/auth/login")
async def login(body: LoginRequest):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, password_hash, display_name, role FROM app_user WHERE email=%s",
                (body.email,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or not row.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = _build_user_payload(row)
    token = create_access_token(user_id=user["id"], email=user["email"])
    return {"user": user, "access_token": token, "token_type": "bearer"}


@router.get("/api/auth/me")
async def me(user_id: str = Depends(require_auth)):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, display_name, role, created_at, updated_at FROM app_user WHERE id=%s::uuid",
                (user_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "display_name": row.get("display_name"),
        "role": row.get("role") or "student",
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
