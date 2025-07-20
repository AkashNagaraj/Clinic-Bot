# Clinic-Bot

## Install requirements
a. ```python -m venv venv```

b. ```source venv/bin/activate```

c. ```pip install -r requirements```

## Install Vector DB
[Milvus Installation] (https://milvus.io/docs/v2.2.x/install_standalone-docker.md)

## Ollama Setup and execution with multiple processes
a. [Ollama Installation] (https://ollama.com/)

b. ```OLLAMA_NUM_PARALLEL=4 ollama serve```

## Run the Backend
a. ```cd BE```

b. Run the tools (currently RAG and simple API)
```python rag_implementations.py```

c. Agent Orchestrator
```python main.py```

## Run FE
a. ```cd FE```

b. ```streamlit run app.py```

c. ```streamlit run dashboard.py```
