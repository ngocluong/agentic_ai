from pydantic_settings import BaseSettings, SettingsConfigDict
import os
ENV_FILE = os.getenv("ENV_FILE", ".env") # set default to be .env
# ENV_FILE=dev.env python -m w7_agent.main reflection

class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=ENV_FILE, enable_decoding=False, extra="ignore")
  
  groq_api_key: str
  llm_model: str = "openai/gpt-oss-120b"
  chroma_path: str = "./chroma"
  max_iterations: int = 3
  temperature: float = 0.0
  tavily_api_key: str
  
settings = Settings()