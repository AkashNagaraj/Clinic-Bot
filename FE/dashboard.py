import streamlit as st
import os
from datetime import datetime
import time

LOG_FILE = "../BE/orchestrator.log"

def read_logs():
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-10:]


def parse_logs(lines):
    parsed = []
    for line in lines:
        try:
            timestamp, message = line.strip().split("|")
            parsed.append({
                "timestamp": timestamp.strip("["),
                "message": message
            })
        except ValueError:
            continue
    return parsed


def show_dashboard():
    st.set_page_config(layout="wide")
    st.title("🛠️ FastAPI Logs Dashboard")

    refresh_interval = 5  # seconds
    last_updated = datetime.now().strftime("%H:%M:%S")

    st.markdown(f"⏰ Last updated: `{last_updated}` &nbsp;&nbsp;&nbsp;&nbsp; 🔁 Auto-refresh every {refresh_interval} sec")

    logs = parse_logs(read_logs())

    for log in logs:
        message = log["message"]
        timestamp = log["timestamp"]
        if "query" in message.lower():
            st.markdown(f"<span style='color:green; font-weight:600;'>🟢 Query</span> {timestamp}<br>➤ {message}", unsafe_allow_html=True)
        elif "response" in message.lower():
            st.markdown(f"<span style='color:blue; font-weight:600;'>🔵 Response</span> {timestamp}<br>💬 {message}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:gray; font-weight:500;'>🔘 Log</span> {timestamp}<br>{message}", unsafe_allow_html=True)

    # Auto-refresh workaround using meta-refresh
    st.markdown(f"""
        <meta http-equiv="refresh" content="{refresh_interval}">
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_dashboard()