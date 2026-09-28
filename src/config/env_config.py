from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Research Copilot"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # LLM MODELS
    PLANNER_MODEL: str
    GENERATOR_MODEL: str

    # API KEYS
    OPEN_ROUTER_API_KEY: SecretStr

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


settings = Settings()  # type: ignore[call-arg]
