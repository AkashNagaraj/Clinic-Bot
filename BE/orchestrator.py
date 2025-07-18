from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from crewai import Agent, Crew, Task, Process, LLM
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

from langchain_redis import RedisChatMessageHistory
from langchain.schema import HumanMessage,AIMessage

from rag_implementations import query_booking_rag

import json

# Initialize Ollama LLM
llm = LLM(
    model="ollama/llama3",          
    temperature=0.2,
    base_url="http://localhost:11434"  # default Ollama endpoint
)

REDIS_URL = "redis://localhost:6379"
def add_to_session(session_name, message_type, message):
    # Use session_name as part of the Redis key
    full_key = f"{session_name}"
    custom_history = RedisChatMessageHistory(full_key, redis_url=REDIS_URL)
    custom_history.add_message({"type":message_type, "message":message})

def add_to_history(agent_type, user_input, agent_output):
    history = {}
    history.add_message(HumanMessage(content=str(user_input)))
    history.add_message(AIMessage(content=str(agent_output)))

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
    prompt_template = f"""
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

    Strictly return with the following Schema : 
    <jsonstart>
    {{
        "class_type":"continue" or "new_task",
    }}
    <jsonend>

    User Query: "{query}"

    """

    intent_task = Task( 
        description = prompt_template, 
        expected_output = "A valid JSON matching the Intent_Classification schema",
        agent = intent_agent,
        output_json = Intent_Classification)

    crew = Crew(agents=[intent_agent], tasks=[intent_task], process=Process.sequential, verbose=False)
    # result = crew.kickoff()
    result = llm.call(prompt_template)
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

        Strictly return with the following Schema :
        <jsonstart>
        {{
            "class_type" : "dental" / "non-dental"
        }}
        <jsonend>
        User Query: "{query}"
        """

    domain_task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=domain_agent)

    crew = Crew(agents=[domain_agent], tasks=[domain_task], process=Process.sequential, verbose=False)
    # result = crew.kickoff()
    result = llm.call(prompt_template)
    return result

def query_clarity(query):
    prompt_template = f"""
        You are a domain classification agent. Your job is to analyze a user's query and classify it as FAQ / Booking/ Other based on the user query.
        ---
        ### Classification Rules:

        Important:
        - Output ONLY the JSON matching the above schema. Do NOT add any commentary or extra text.

        Strictly return with the following Schema : - 
        <jsonstart>
        {{
            "class_type" : "FAQ" / "Booking" / "Vague"
        }}
        <jsonend>
        User Query: "{query}"
        """

    domain_task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=query_clarity_agent)

    crew = Crew(agents=[query_clarity_agent], tasks=[domain_task], process=Process.sequential, verbose=False)
    # result = crew.kickoff()
    result = llm.call(prompt_template)
    return result

def answer(query, prev_context = []):
    summary_input = [context["result"] for context in prev_context]
    # summary_input = json.dumps(prev_context, indent=2, ensure_ascii=False)
    prompt_template = f"""
    You are a helpful and knowledgeable healthcare assistant.

    Your task is to read the user's query and use the prior context to generate a clear, relevant, and concise response around 10-25 words. You must extract the most important information from the context that directly relates to the current query and summarize it appropriately.

    ### User Query:
    {query}

    ### Prior Context:
    {summary_input}

    Based on the above, return an appropriate response.
    """

    result = llm.call(prompt_template)
    return result

# Request model
class QueryInput(BaseModel):
    query: str

def preprocess_text(query):
    import re
    match = re.search(r"<\/?jsonstart>\s*(\{.*?\})\s*<\/?jsonend\/?>", query, re.DOTALL)
    if match:   
        json_data = match.group(1)
    json_data = json.loads(json_data)
    return json_data
 
@app.post("/query")
async def process_query(data: QueryInput):
    previous_context = []
    continue_check = preprocess_text(classify_intent(data.query))

    # if continue_check["class_type"]=="new_task":
    #     dump_to_redis()
    #     previous_context = []
        
    # domain_class = preprocess_text(classify_domain(data.query))

    query_details = preprocess_text(query_clarity(data.query))
    print("Query Details : ", query_details)
    if query_details["class_type"].lower() == "faq":
        rag_response = query_booking_rag(data.query)
        print("FAQ RAG : ",rag_response)
        previous_context.append({"agent_type":"query_clarity_agent", "query":data.query, "result":rag_response})
    elif query_details["class_type"].lower() == "booking":
        # call booking api
        previous_context.append({"agent_type":"query_clarity_agent", "query":data.query, "result":"The available slots are morning and night"})        
    elif query_details["class_type"].lower() == "vague":
        previous_context.append({"agent_type":"query_clarity_agent", "query":data.query, "result":"The query seems vague could you please elaborate."}) 
    else:
        previous_context.append({"agent_type":"domain_agent", "query":data.query, "result":"Please ask a question relating to Booking/ FAQ"})
    
    final_response = answer(data.query, previous_context)
    previous_context.append({"agent_type":"answer_agent", "query":data.query, "result":final_response})

    return {"response": final_response}

if __name__=="__main__":
    uvicorn.run("orchestrator:app", host="127.0.0.1", port=8030, reload=True)

