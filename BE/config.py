from crewai import LLM

REDIS_URL = "redis://localhost:6379"
llm = LLM(
    model="ollama/llama3",
    temperature=0.2,
    base_url="http://localhost:11434"
)