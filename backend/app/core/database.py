from supabase import create_client, Client
from app.core.config import get_settings


def get_supabase_client() -> Client:
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


# Singleton client
_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = get_supabase_client()
    return _client
