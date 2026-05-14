"""配置管理。pydantic-settings 同时支持 yaml + 环境变量。"""
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelsConfig(BaseModel):
    master_agent: str
    sub_agent: str
    recorder: str
    reviewer: str


class LLMConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 1.0
    reviewer_temperature: float = 0.3
    request_timeout_s: int = 120
    max_retries: int = 3


class RetrievalConfig(BaseModel):
    auto_retrieve_n: int = 5
    verbatim_max_results: int = 5


class ReviewLoopConfig(BaseModel):
    similarity_threshold: float = 0.85
    window: int = 3
    max_no_progress: int = 3


class StorageConfig(BaseModel):
    data_dir: Path = Path("./.data")
    output_dir: Path = Path("./outputs")
    chroma_dir: Path = Path("./.data/chroma")
    trace_dir: Path = Path("./.data/traces")


class ConcurrencyConfig(BaseModel):
    sub_agent_parallel: bool = True


class ProviderConfig(BaseModel):
    type: str = "anthropic"  # anthropic | openai
    base_url: str = "https://sub.whitedream.top/v1"
    api_key: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore",
    )
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    provider: ProviderConfig = ProviderConfig()
    models: ModelsConfig
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    review_loop: ReviewLoopConfig = ReviewLoopConfig()
    storage: StorageConfig = StorageConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    prompts_dir: Path = Path("./prompts")

    @classmethod
    def load(cls, yaml_path: str = "config.yaml") -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def ensure_dirs(self):
        for p in (self.storage.data_dir, self.storage.output_dir,
                  self.storage.chroma_dir, self.storage.trace_dir):
            p.mkdir(parents=True, exist_ok=True)