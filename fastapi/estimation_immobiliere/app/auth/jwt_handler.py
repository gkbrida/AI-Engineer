from decouple import config 
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt


secret = config("secret_key_jwt")
algorithm = config("algorithm_jwt")
expire = config("expire_jwt")

# Functio return generated token
def token_response (token:str):
    return{
        "access token" : token 
    }
    
def creat_jwt(data : dict):
    payload = data.copy()
    exp = datetime.now(timezone.utc) + timedelta(minutes=int(expire))
    payload["expire"] = int(exp.timestamp())
    token = jwt.encode(payload, secret, algorithm)
    return token

def decode_jwt(token:str):
    try: 
        decode_token = jwt.decode(token, secret, algorithm)
        if decode_token.expire < datetime.now(timezone.utc):
            return None
        return decode_token  
    except:
        return {}
    
    
    