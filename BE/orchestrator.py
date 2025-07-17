from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from crewai import Agent, Crew, Task, Process, LLM
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

# from crewai.llms import Ollama

# Initialize Ollama LLM
llm = LLM(
    model="ollama/llama3",          
    temperature=0.2,
    base_url="http://localhost:11434"  # default Ollama endpoint
)

app = FastAPI()


def extract_json(text):
    """Extract the first JSON object from the given text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in output.")
    return match.group(0)


# Intent (Continue or Not)
class Intent_Classification(BaseModel):
    class_type: Literal['continue', 'new_task']
class DomainClassification(BaseModel):
    class_type: Literal['dental', 'non-dental']
class QueryClarity(BaseModel):
    class_type: Literal["FAQ", "Booking", "Other"]

intent_agent = Agent(
    role="Classification Agent",
    goal="Understand user query and identify if the intent is create a new task or answer from a previous question",
    backstory="""You are an LLM-powered reasoning agent responsible for answering new or continue.""",
    verbose=False,
    allow_delegation=False,
    llm=llm,
    reasoning=False,
    max_reasoning_attempts=2
    )
domain_agent = Agent(
    role="Domain Classifier",
    goal="Classify the user query as related to dental or non-dental",
    backstory="You are an LLM-powered reasoning agent responsible for answering dental or other.",
    verbose=False,
    allow_delegation=False,
    llm=llm,
    reasoning=False,
    max_reasoning_attempts=2
)
query_clarity_agent = Agent(
    role="Query Classifier",
    goal="Classify the user query as related to FAQ or Booking",
    backstory="You are an LLM-powered reasoning agent responsible for answering if the query is about FAQ or Bookings or Other.",
    verbose=False,
    allow_delegation=False,
    llm=llm,
    reasoning=False,
    max_reasoning_attempts=2
)

def classify_intent(query):
    intent_task = Task( 
    description = f"""
    You are an intent classification agent. Your job is to analyze a user's natural language query and classify it into one of two task types based on whether it represents a new request or a continuation of a previous task.
    ---

    ### Classification Rules:

    Use the following strict criteria to determine the correct `class_type`:

    - **'continue'**:
    - The input refers to or depends on a previous message or result.
    - Common indicators: "summarize that", "send this", "explain above", "what does that mean", "do it again", "continue"
    - Typically lacks full task context and relies on conversation memory.

    - **'new_task'**:
    - The input is self-contained and clearly starts a new request or topic.
    - Includes its own subject or goal 

    Important:
    - If the intent clearly starts a new actionable or informational task, classify as `'new_task'`.
    - If the intent is conversational or meta-linguistic (referring to prior interaction), classify as `'continue'`.

    Return only the JSON output with no explanation.

    User Query: "{query}"

    """, 
    expected_output = "A valid JSON matching the Intent_Classification schema",
    agent = intent_agent,
    output_json = Intent_Classification)

    crew = Crew(agents=[intent_agent], tasks=[intent_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return result

def classify_domain(query):
    prompt_template = f"""
        You are a domain classification agent. Your job is to analyze a user's query and classify it as dental or non-dental based on the user query.
        ---
        ### Classification Rules:

        Use the following strict criteria to determine the correct `class_type`:
        - **'dental'**:
            - Refers to anything involving teeth, gums, dentists, dental hygiene, braces, extractions, cleaning, etc.
            - Examples: "Book a dental cleaning", "I have a toothache", "I need a dentist appointment"

        - **'non-dental'**:
            - Anything that is NOT directly related to dentistry.
            - Could include skin care, eye checkups, general physician questions, etc.

        Important:
        - Output ONLY the JSON matching the above schema. Do NOT add any commentary or extra text.

        Schema -
        {{
            "class_type" : "dental" / "non-dental"
        }}

        User Query: "{query}"
        """

    domain_task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=domain_agent)

    crew = Crew(agents=[domain_agent], tasks=[domain_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    result = llm.call(prompt_template)
    return result

def query_clarity(query):
    prompt_template = f"""
        You are a domain classification agent. Your job is to analyze a user's query and classify it as FAQ / Booking/ Other based on the user query.
        ---
        ### Classification Rules:

        Important:
        - Output ONLY the JSON matching the above schema. Do NOT add any commentary or extra text.

        Schema -
        {{
            "class_type" : "FAQ" / "Booking" / "Other"
        }}

        User Query: "{query}"
        """

    domain_task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=query_clarity_agent)

    crew = Crew(agents=[query_clarity_agent], tasks=[domain_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    result = llm.call(prompt_template)
    return result

def answer(query, prev_context = ""):
    prompt_template = f"""
    You are a healthcare agent. Your task is to answer the query using the previous context.
    Return the answer in the following schema:

    User Query - 
    {query}

    Context - 
    {prev_context}

    Schema - 
    {{
        "response": "answer"
    }}
    """
    result = llm.call(prompt_template)
    return result

# Request model
class QueryInput(BaseModel):
    query: str


@app.post("/query")
async def process_query(data: QueryInput):
    continue_check = classify_intent(data.query)
    domain_class = classify_domain(data.query)
    query_details = query_clarity(data.query)
    final_response = answer(data.query)
    return {"response": final_response}

if __name__=="__main__":
    uvicorn.run("orchestrator:app", host="127.0.0.1", port=8030, reload=True)

