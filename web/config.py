from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_name: str = "postgres"
    db_user: str = "postgres"
    db_password: str = "yoursecretpassword"
    db_host: str = "localhost"
    ollama_url: HttpUrl = HttpUrl("http://ollama:11434")

    # Holds the path to the secret file; not exposed outside this class
    db_password_file: str | None = None

    @model_validator(mode="after")
    def load_password_from_file(self) -> "Settings":
        if self.db_password_file:
            secret = Path(self.db_password_file).read_text().strip()
            self.db_password = secret
        return self

    ollama_model_embedding_selected: str
    ollama_model_embedding_principal: str
    ollama_model_embedding_ablation: str
    ollama_model_llm_generador: str
    top_k_selected: int = Field(3, ge=1)
    chunk_selected: Literal["full"] | Literal["sliding"]
    ollama_num_ctx: int = 2048
    temperature_selected: float = Field(0.0, ge=0, le=1)

    # model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()
