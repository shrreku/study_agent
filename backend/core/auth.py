"""Auth utilities and dependencies for FastAPI.

- security: HTTPBearer instance
- require_auth(): dependency that validates Authorization header
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    # Placeholder: accept any Bearer token for MVP (validated presence only)
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth")
    return credentials.credentials
