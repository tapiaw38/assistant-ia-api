import aiohttp
import urllib.parse


def get_mercadolibre_auth_url(app_id: str, redirect_uri: str) -> str:
    base_url = "https://auth.mercadolibre.com.ar/authorization"
    params = {
        "response_type": "code",
        "client_id": app_id,
        "redirect_uri": redirect_uri
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def extract_code_from_redirect(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    return query.get("code", [None])[0]

async def get_mercadolibre_token(app_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    """
    Intercambia el code por el access_token y user_id usando la API de MercadoLibre.
    Retorna el JSON de respuesta (incluye access_token, user_id, refresh_token, etc).
    """
    url = "https://api.mercadolibre.com/oauth/token"
    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "client_id": app_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            return await response.json()
