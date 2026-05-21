import os
import base64
import requests
import webbrowser
from loguru import logger

def main():
    app_key = "2jKcXEUV7aoAblcJV7RSDuDGjAslUjAL" 
    app_secret = "Gf4IG7DTIpc0FHqF"

    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri=https://127.0.0.1"

    logger.info("Click to authenticate:")
    logger.info(auth_url)
    webbrowser.open(auth_url)
    logger.info("Paste Returned URL:")
    returned_url = input()

    response_code = f"{returned_url[returned_url.index('code=') + 5: returned_url.index('%40')]}@"
    credentials = f"{app_key}:{app_secret}"
    base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {base64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    payload = {
        "grant_type": "authorization_code",
        "code": response_code,
        "redirect_uri": "https://127.0.0.1",
    }

    init_token_response = requests.post(
        url="https://api.schwabapi.com/v1/oauth/token",
        headers=headers,
        data=payload,
    )

    init_tokens_dict = init_token_response.json()
    logger.debug(init_tokens_dict)
    return "Done!"


if __name__ == "__main__":
    main()

