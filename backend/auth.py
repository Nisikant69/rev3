import time, jwt, os, requests
from config import GITHUB_APP_ID, GITHUB_PRIVATE_KEY 

APP_ID, PRIVATE_KEY = GITHUB_APP_ID, GITHUB_PRIVATE_KEY
def build_app_jwt():
    now = int(time.time())
    payload = {"iat": now, "exp": now + 9 * 60, "iss": APP_ID}
    private_key_str = PRIVATE_KEY
    if PRIVATE_KEY and not PRIVATE_KEY.strip().startswith("-----BEGIN"):
        with open(PRIVATE_KEY.strip(), "r") as f:
            private_key_str = f.read()
    return jwt.encode(payload, private_key_str.encode("utf-8"), algorithm="RS256")

def get_installation_token(installation_id: str) -> str:
    app_jwt = build_app_jwt()
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()["token"]
