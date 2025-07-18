import uvicorn
from fastapi import FastAPI
from models import QueryInput
from config import llm
from agents import intent_agent, domain_agent, query_clarity_agent
from tasks import classify_intent, query_clarity, answer
from utils import preprocess_text, query_booking_rag, get_booking_data

app = FastAPI()

# User context store
user_context_store = {}

@app.post("/query")
async def process_query(data: QueryInput):
    user_id = data.user_id
    query = data.query

    # Get or initialize context
    previous_context = user_context_store.get(user_id, [])

    # Check if this is a continuation or new task
    continue_check = preprocess_text(classify_intent(query, intent_agent, llm))
    if continue_check["class_type"] == "new_task":
        previous_context = []  # Clear context on new task

    # Process the query using classification agent
    query_details = preprocess_text(query_clarity(query, query_clarity_agent, llm))
    print("Query Details:", query_details)

    if query_details["class_type"].lower() == "faq":
        rag_response = query_booking_rag(query)
        print("FAQ RAG:", rag_response)
        previous_context.append({
            "agent_type": "query_clarity_agent",
            "query": query,
            "result": rag_response
        })
    elif query_details["class_type"].lower() == "booking":
        booking_data = get_booking_data()
        previous_context.append({
            "agent_type": "query_clarity_agent",
            "query": query,
            "result": booking_data
        })
    elif query_details["class_type"].lower() == "vague":
        previous_context.append({
            "agent_type": "query_clarity_agent",
            "query": query,
            "result": "The query seems vague. Could you please elaborate?"
        })
    else:
        previous_context.append({
            "agent_type": "domain_agent",
            "query": query,
            "result": "Please ask a question related to Booking/FAQ"
        })

    # Final answer generation
    final_response = answer(query, previous_context, llm)
    previous_context.append({
        "agent_type": "answer_agent",
        "query": query,
        "result": final_response
    })

    # Update context store
    user_context_store[user_id] = previous_context

    return {"response": final_response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8030, reload=True)