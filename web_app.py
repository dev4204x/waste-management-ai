import os
import json
import math
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="centered"
)

LLM_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

MEMORY_FILE = "memory.json"
REPORTS_FILE = "reports.json"

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY was not found. Set your API key in PowerShell, "
        "then restart Streamlit."
    )
    st.stop()


# ---------------- GEMINI CLIENT ----------------

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


# ---------------- WASTE KNOWLEDGE BASE ----------------

DOCUMENTS = [
    {
        "title": "Plastic bottles",
        "text": (
            "Empty plastic bottles before disposal. Where local rules allow, "
            "rinse and place recyclable bottles in the designated dry-waste "
            "or plastic recycling collection. Follow local municipal rules."
        )
    },
    {
        "title": "Food scraps",
        "text": (
            "Fruit peels, vegetable scraps and other suitable food waste can "
            "often go into a wet-waste or composting collection. Keep them "
            "separate from dry recyclables and follow local rules."
        )
    },
    {
        "title": "Batteries and e-waste",
        "text": (
            "Do not put batteries in ordinary household waste or burn them. "
            "Keep damaged batteries away from heat and avoid handling leaking "
            "materials. Use an authorised battery collection point or "
            "e-waste collection service."
        )
    },
    {
        "title": "Paper and cardboard",
        "text": (
            "Keep paper and cardboard clean and dry for recycling. Separate "
            "them from food waste. Greasy or heavily contaminated paper may "
            "not be recyclable through ordinary paper collection."
        )
    },
    {
        "title": "Broken glass",
        "text": (
            "Broken glass can cause injuries. Do not put loose shards into "
            "a bag that someone could handle unsafely. Carefully contain and "
            "label the material, and check your local waste collection rules."
        )
    },
    {
        "title": "Sanitary waste",
        "text": (
            "Wrap sanitary waste securely and keep it separate from ordinary "
            "recyclables. Follow local municipal instructions for collection "
            "and disposal. Do not flush sanitary products down the toilet."
        )
    }
]


def get_embedding(text, task_type):
    response = get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return response.embeddings[0].values


@st.cache_resource
def get_knowledge_base():
    embedded_documents = []

    for document in DOCUMENTS:
        vector = get_embedding(
            document["text"],
            "RETRIEVAL_DOCUMENT"
        )

        embedded_documents.append({
            **document,
            "embedding": vector
        })

    return embedded_documents


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def retrieve_documents(query, top_k=3):
    knowledge_base = get_knowledge_base()
    query_vector = get_embedding(query, "RETRIEVAL_QUERY")

    ranked_documents = []

    for document in knowledge_base:
        score = cosine_similarity(
            query_vector,
            document["embedding"]
        )

        ranked_documents.append((score, document))

    ranked_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        {
            "title": document["title"],
            "text": document["text"],
            "similarity": round(score, 3)
        }
        for score, document in ranked_documents[:top_k]
    ]


# ---------------- AI TOOLS ----------------

def lookup_disposal_guidance(query: str) -> dict:
    """Retrieve relevant waste-disposal guidance from the knowledge base."""

    results = retrieve_documents(query)

    return {
        "results": results,
        "note": (
            "These are general guidelines. Local municipal rules may differ."
        )
    }


def create_sanitation_report(
    location: str,
    issue_description: str
) -> dict:
    """Save a sanitation complaint draft locally; do not submit it externally."""

    report = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "location": location,
        "issue_description": issue_description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "Draft - not submitted"
    }

    try:
        if os.path.exists(REPORTS_FILE):
            with open(REPORTS_FILE, "r", encoding="utf-8") as file:
                reports = json.load(file)
        else:
            reports = []

        reports.append(report)

        with open(REPORTS_FILE, "w", encoding="utf-8") as file:
            json.dump(reports, file, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "report": report,
            "message": (
                "The complaint draft was saved locally. "
                "It has NOT been submitted to any government authority."
            )
        }

    except (OSError, json.JSONDecodeError) as error:
        return {
            "success": False,
            "message": f"Could not save the report draft: {error}"
        }


# ---------------- MEMORY ----------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

        except (OSError, json.JSONDecodeError):
            pass

    return []


def save_memory(history):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history[-12:], file, indent=2, ensure_ascii=False)
    except OSError:
        st.warning("Conversation memory could not be saved to disk.")


def convert_history_to_gemini(history):
    contents = []

    for message in history[-12:]:
        role = message.get("role")
        content = message.get("content", "")

        if role not in ("user", "model") or not content:
            continue

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=content)]
            )
        )

    return contents


# ---------------- SYSTEM INSTRUCTIONS ----------------

SYSTEM_INSTRUCTION = """
You are WasteWise AI, a helpful waste-management assistant.

Rules:
1. For waste-disposal questions, use lookup_disposal_guidance to retrieve
   relevant information before answering.
2. Explain the retrieved guidance clearly and simply.
3. Retrieved guidance is general information, not a guarantee of local rules.
   Never invent municipal rules, collection locations, phone numbers, or
   official services.
4. If local rules matter, ask the user for their city or municipality.
5. You can create a local sanitation complaint draft using
   create_sanitation_report.
6. Before creating a draft, obtain clear permission from the user.
7. If the location or issue description is missing, ask for it first.
8. Clearly explain that drafts are saved locally and are NOT submitted to
   a government authority.
9. Use the conversation history to understand follow-up questions.
10. Do not claim that a complaint was officially submitted.
11. Be friendly, concise, and practical.
"""


CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[
        lookup_disposal_guidance,
        create_sanitation_report
    ]
)


# ---------------- WEBSITE INTERFACE ----------------

st.title("♻️ WasteWise AI")
st.caption(
    "Your AI assistant for waste disposal and sanitation reporting."
)

with st.sidebar:
    st.header("About WasteWise")
    st.write(
        "Ask about waste segregation, recycling, batteries, "
        "food waste, and sanitation issues."
    )

    st.info(
        "This prototype uses the Gemini API and a small demonstration "
        "knowledge base. Disposal rules can vary by municipality."
    )

    if st.button("Clear conversation memory", use_container_width=True):
        st.session_state.history = []
        save_memory([])
        st.session_state.chat = get_client().chats.create(
            model=LLM_MODEL,
            config=CONFIG
        )
        st.rerun()


# Prepare the knowledge base once and cache its embeddings.
with st.spinner("Preparing the waste knowledge base..."):
    try:
        get_knowledge_base()
    except Exception as error:
        st.error(
            "Could not prepare the knowledge base. Check your Gemini API "
            "key, internet connection, API access, and available quota."
        )
        st.code(str(error))
        st.stop()


# Load persistent memory once per browser session.
if "history" not in st.session_state:
    st.session_state.history = load_memory()[-12:]


# Create a Gemini chat with the saved text history.
if "chat" not in st.session_state:
    try:
        saved_history = convert_history_to_gemini(
            st.session_state.history
        )

        st.session_state.chat = get_client().chats.create(
            model=LLM_MODEL,
            config=CONFIG,
            history=saved_history
        )

    except Exception as error:
        st.error("Could not start the Gemini chat.")
        st.code(str(error))
        st.stop()


# Display previous messages.
for message in st.session_state.history:
    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "model":
        with st.chat_message("assistant"):
            st.markdown(content)


# Chat input.
user_message = st.chat_input(
    "Ask about waste disposal or sanitation..."
)

if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)

    st.session_state.history.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("assistant"):
        with st.spinner("WasteWise is thinking..."):
            try:
                response = st.session_state.chat.send_message(
                    user_message
                )

                answer = response.text

                if not answer:
                    answer = (
                        "I couldn't generate a text response. "
                        "Please try asking in a different way."
                    )

            except Exception as error:
                answer = (
                    "I encountered an error while contacting Gemini. "
                    "Check your API quota, internet connection, and "
                    "API configuration, then try again."
                )
                st.error(str(error))

        st.markdown(answer)

    st.session_state.history.append({
        "role": "model",
        "content": answer
    })

    st.session_state.history = st.session_state.history[-12:]
    save_memory(st.session_state.history)