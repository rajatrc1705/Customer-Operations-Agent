from typing import Any

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: str | None = None
    agent_model: str = "gpt-5-nano"


def create_model(settings: Settings | None = None) -> Any:
    configured = settings or Settings()

    if not configured.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
            "Copy .env.example to .env and add your API key."
        )

    return ChatOpenAI(
        model=configured.agent_model,
        api_key=configured.openai_api_key,
        use_responses_api=True
    )