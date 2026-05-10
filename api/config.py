from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider selection — "openrouter" | "ollama" | "vllm"
    llm_provider: str = "openrouter"

    # Pipeline selection — "single" | "two_step" | "two_step_all" | "three_step"
    reply_pipeline: str = "single"

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

    # single pipeline
    system_prompt_file: str = "training/prompts/single/system_v4.txt"

    # two_step pipeline
    two_step_brainstorm_file: str = "training/prompts/two_step/step1_brainstorm_v1.txt"
    two_step_refiner_file: str = "training/prompts/two_step/step2_refiner_v1.txt"
    two_step_refiner_all_file: str = "training/prompts/two_step/step2_refiner_all_v1.txt"

    # three_step pipeline
    three_step_analyze_file: str = "training/prompts/three_step/step1_analyze_v1.txt"
    three_step_evaluate_file: str = "training/prompts/three_step/step2_evaluate_v1.txt"
    three_step_deliver_file: str = "training/prompts/three_step/step3_deliver_v1.txt"
    # Step 3 reply model — swap to "qwen-local" when Ollama SFT is ready
    three_step_reply_model: str = "gpt-4o-mini"

    # Judge / shared
    judge_model: str = "openai/gpt-4o-mini"
    judge_prompt_file: str = "training/prompts/judge_v1.txt"

    # App settings
    db_path: str = "chatbot.db"
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
