"""Hugging Face authentication helpers for Kaggle, Colab, and local runs."""

from __future__ import annotations

import os
from typing import Optional


def load_hf_token(secret_name: str = "HF_TOKEN") -> Optional[str]:
    """Load a Hugging Face token from env, Kaggle Secrets, or Colab Secrets."""

    token = os.getenv(secret_name, "").strip() or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient

        token = (UserSecretsClient().get_secret(secret_name) or "").strip()
        if token:
            return token
    except Exception:
        pass

    try:
        from google.colab import userdata

        token = (userdata.get(secret_name) or "").strip()
        if token:
            return token
    except Exception:
        pass

    return None


def configure_huggingface_auth(secret_name: str = "HF_TOKEN", login: bool = True) -> bool:
    """Configure HF Hub auth without exposing the token in notebook output.

    Returns ``True`` when a token was found and exported, otherwise ``False``.
    """

    token = load_hf_token(secret_name=secret_name)
    if not token:
        print("HF_TOKEN not found. Continuing unauthenticated; downloads may be rate limited.")
        return False

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    if login:
        try:
            from huggingface_hub import login as hf_login

            hf_login(token=token, add_to_git_credential=False)
        except Exception as error:
            print(f"HF token loaded, but huggingface_hub.login() failed: {error}")
            return True

    print("HF token loaded.")
    return True
