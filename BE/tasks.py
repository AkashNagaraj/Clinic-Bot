from crewai import Task, Crew, Process
from models import Intent_Classification

def classify_intent(query, agent, llm):
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
    task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the Intent_Classification schema",
        agent=agent,
        output_json=Intent_Classification
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return llm.call(prompt_template)

def classify_domain(query, agent, llm):
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
    task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbosemel=False)
    return llm.call(prompt_template)

def query_clarity(query, agent, llm):
    prompt_template = f"""
    You are a domain classification agent. Your job is to analyze a user's query and classify it as FAQ / Booking / Other based on the user query.
    ---
    ### Classification Rules:

    Important:
    - Output ONLY the JSON matching the above schema. Do NOT add any commentary or extra text.

    Strictly return with the following Schema :
    <jsonstart>
    {{
        "class_type" : "FAQ" / "Booking" / "Vague"
    }}
    <jsonend>
    User Query: "{query}"
    """
    task = Task(
        description=prompt_template,
        expected_output="A valid JSON matching the schema",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return llm.call(prompt_template)

def answer(query, prev_context, llm):
    summary_input = [context["result"] for context in prev_context]
    prompt_template = f"""
    You are a helpful and knowledgeable healthcare assistant.

    Your task is to read the user's query and use the prior context to generate a clear, relevant, and concise response around 10-25 words. You must extract the most important information from the context that directly relates to the current query and summarize it appropriately.

    ### User Query:
    {query}

    ### Prior Context:
    {summary_input}

    Based on the above, return an appropriate response.
    """
    return llm.call(prompt_template)