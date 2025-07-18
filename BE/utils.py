import json
import re
import requests
from langchain_redis import RedisChatMessageHistory
from config import REDIS_URL

def add_to_session(session_name, message_type, message):
    full_key = f"{session_name}"
    custom_history = RedisChatMessageHistory(full_key, redis_url=REDIS_URL)
    custom_history.add_message({"type": message_type, "message": message})

def preprocess_text(query_response):
    match = re.search(r"</?jsonstart>\s*(\{.*?\})\s*</?jsonend/?>", query_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in model response.")
    return json.loads(match.group(1))

def query_booking_rag(query):
    url = "http://localhost:8040/query_faq"
    payload = {"query": query}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("matches", [])
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return []

def get_booking_data():
    url = "http://localhost:8040/bookings"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("message")
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return {}