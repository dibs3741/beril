import jwt
from rich.console import Console
from jwt.exceptions import InvalidTokenError
from fastapi import FastAPI, Depends, HTTPException,status
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import TokenTable

ALGORITHM = "HS256"
JWT_SECRET_KEY = "secret-key"   # should be kept secret
console = Console()

def decodeJWT(jwtoken: str):
    console.log("decodeJWT")
    try:
        # Decode and verify the token
        payload = jwt.decode(jwtoken, JWT_SECRET_KEY, ALGORITHM)
        return payload
    except InvalidTokenError:
        return None


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        console.log("initializing JWTBearer")
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        console.log("jwtbearer in action")
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        console.log("jwtbearer in action 1")
        if credentials:
            console.log("jwtbearer in action 2")
            if not credentials.scheme == "Bearer":
                console.log("Invalid authentication scheme.")
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                console.log("Invalid token or expired token.")
                raise HTTPException(status_code=403, detail="Invalid token or expired token.")
            return credentials.credentials
        else:
            console.log("Invalid authorization code")
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwtoken: str) -> bool:
        console.log("verify_jwt")
        isTokenValid: bool = False

        try:
            payload = decodeJWT(jwtoken)
        except:
            payload = None
        if payload:
            isTokenValid = True
        console.log("verify_jwt exit")
        return isTokenValid

jwt_bearer = JWTBearer()
