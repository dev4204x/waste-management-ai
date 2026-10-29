
import os
import json
import math
from datetime import datetime
from google import genai
from google.genai import types

# ---------- CONFIGURATION ----------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY is missing. Set it in your terminal first."
    )

client = genai.Client(api_key=API_KEY)

LLM_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

MEMORY_FILE = "memory.json"
REPORTS_FILE = "reports.json"


# ---------- RAG: KNOWLEDGE BASE ----------

DOCUMENTS = [
    {
        "topic": "Plastic bottles",
        "text": (
            "Clean plastic bottles may be recyclable. Keep them separate "
            "from wet waste and follow the local municipal recycling rules."
        )
    },
    {
        "topic": "Food scraps",
        "text": (
            "Fruit peels, vegetable scraps and similar food waste may be "
            "suitable for wet-waste collection or composting where available."
        )
    },
    {
        "topic": "Batteries",
        "text": (
            "Do not mix used batteries with ordinary household waste. "
            "Keep them separate and use an authorized collection point "
            "or an applicable take-back program."
        )
    },
    {
        "topic": "Paper and cardboard",
        "text": (
            "Keep recyclable paper and cardboard dry and reasonably clean. "
            "Follow local collection rules for recyclable dry waste."
        )
    },
    {
        "topic": "Broken glass",
        "text": (
            "Wrap broken glass securely in a rigid container, label it "
            "clearly if possible, and follow local disposal instructions "
            "to protect waste handlers."
        )
    },
    {
        "topic": "Sanitary waste",
        "text": (
            "Wrap sanitary waste securely and follow local municipal "
            "collection and disposal guidance. Do not place it loose "
            "with recyclable materials."
        )
    }
]


def get_embedding(text, task_type):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )
    return result.embeddings[0].values


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


print("Preparing the waste knowledge base...")
for document in DOCUMENTS:
    document["embedding"] = get_embedding(
        document["text"],
        "RETRIEVAL_DOCUMENT"
    )
print("Knowledge base ready.")


def retrieve_documents(query, top_k=3):
    query_embedding = get_embedding(
        query,
        "RETRIEVAL_QUERY"
    )

    ranked = []

    for document in DOCUMENTS:
        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )
        ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "topic": document["topic"],
            "text": document["text"],
            "similarity": round(score, 3)
        }
        for score, document in ranked[:top_k]
    ]


# ---------- TOOL 1: RAG SEARCH ----------

def lookup_disposal_guidance(query: str) -> dict:
    """Search the waste knowledge base for relevant disposal guidance.

    Args:
        query: Describe the waste item or disposal question.

    Returns:
        Relevant knowledge base passages for answering the question.
    """
    results = retrieve_documents(query)

    return {
        "results": results,
        "notice": (
            "These are general guidance examples, not verified local rules. "
            "Ask for the municipality or local guidelines when necessary."
        )
    }


# ---------- TOOL 2: DRAFT A COMPLAINT ----------

def create_sanitation_report(
    location: str,
    issue_description: str
) -> dict:
    """Save a local draft about a sanitation or illegal dumping issue.

    Args:
        location: User-provided location of the issue.
        issue_description: User-provided description of the problem.

    Returns:
        Confirmation that a local draft was saved. It is not submitted
        to any government or municipal authority.
    """
    if not location.strip() or not issue_description.strip():
        return {
            "error": "A location and issue description are required."
        }

    try:
        with open(REPORTS_FILE, "r", encoding="utf-8") as file:
            reports = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        reports = []

    report = {
        "id": len(reports) + 1,
        "location": location.strip(),
        "issue_description": issue_description.strip(),
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "Draft - not submitted"
    }

    reports.append(report)

    with open(REPORTS_FILE, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)

    return {
        "message": "Sanitation report draft saved locally.",
        "report": report,
        "important": (
            "This draft has not been sent to any authority."
        )
    }


# ---------- MEMORY ----------

def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_memory(history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


history = load_memory()


# ---------- GEMINI AGENT ----------

SYSTEM_INSTRUCTION = """
You are a helpful Waste Management Assistant.

You can:
1. Search the waste knowledge base using lookup_disposal_guidance.
2. Draft a sanitation report using create_sanitation_report.

Rules:
- Use the knowledge base when answering disposal questions.
- Do not invent municipal rules, nearby collection points, or official links.
- Explain when guidance is general and local rules may differ.
- Ask a follow-up question if key information is missing.
- Before drafting a report, obtain the user's clear permission.
- Ask for the location and issue description before drafting.
- A saved report is only a local draft, never an official submission.
- Use the user's conversation context when it is relevant.
- Never claim to know personal information that was not provided.
- Be concise, practical, and respectful.
"""

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[
        lookup_disposal_guidance,
        create_sanitation_report
    ]
)


def build_contents(history, current_message):
    contents = []

    for item in history[-12:]:
        contents.append(
            types.Content(
                role=item["role"],
                parts=[types.Part.from_text(text=item["text"])]
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=current_message)]
        )
    )

    return contents


print("\n===================================")
print(" WASTE MANAGEMENT AI ASSISTANT")
print(" Gemini + RAG + Tools + Memory")
print("===================================")
print("Commands: /exit, /clear")
print("Memory is stored locally in memory.json.")


while True:
    message = input("\nYou: ").strip()

    if message.lower() == "/exit":
        print("Goodbye!")
        break

    if message.lower() == "/clear":
        history = []
        save_memory(history)
        print("Conversation memory cleared.")
        continue

    if not message:
        continue

    try:
        contents = build_contents(history, message)

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=CONFIG
        )

        answer = response.text or (
            "I couldn't produce a text answer. Please try again."
        )

        print("\nAssistant:", answer)

        history.append({
            "role": "user",
            "text": message
        })
        history.append({
            "role": "model",
            "text": answer
        })

        # Keep the last 12 messages to limit stored history.
        history = history[-12:]
        save_memory(history)

    except Exception as error:
        print("\nSomething went wrong:", error)
        print("Check the error message and try again.")
