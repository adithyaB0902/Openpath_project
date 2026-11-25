"""Configuration loaded from environment variables or a .env file."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    github_token: str
    hf_token: str
    db_path: str
    embedder: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str


def get_settings() -> Settings:
    return Settings(
        github_token=os.getenv("GITHUB_TOKEN", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
        db_path=os.getenv("OPENPATH_DB", "data/openpath.db"),
        embedder=os.getenv("OPENPATH_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
        llm_provider=os.getenv("OPENPATH_LLM", "none"),
        llm_model=os.getenv("OPENPATH_LLM_MODEL", ""),
        llm_base_url=os.getenv("OPENPATH_LLM_BASE_URL", ""),
        llm_api_key=os.getenv("OPENPATH_LLM_API_KEY", ""),
    )
