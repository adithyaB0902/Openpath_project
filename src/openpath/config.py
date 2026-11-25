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


def get_settings() -> Settings:
    return Settings(
        github_token=os.getenv("GITHUB_TOKEN", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
        db_path=os.getenv("OPENPATH_DB", "data/openpath.db"),
    )
