from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider selection — "openrouter" | "ollama" | "vllm"
    llm_provider: str = "openrouter"

    # System prompt file (shared across all providers)
    system_prompt_file: str = "training/prompts/system_v2.txt"

    # 1) OpenRouter settings (LLM_PROVIDER=openrouter)
    openrouter_api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "openai/gpt-4o-mini"

    # 2) Ollama settings (LLM_PROVIDER=ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen-emotional-coach"

    # 3) vLLM settings (LLM_PROVIDER=vllm)
    vllm_base_url: str = ""
    vllm_api_key: str = ""
    vllm_model: str = ""

    # App settings
    db_path: str = "chatbot.db"
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
