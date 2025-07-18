from crewai import Agent
from config import llm

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