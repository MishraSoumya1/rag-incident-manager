from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from app.config import CHROMA_DIR, MODEL_NAME, GROQ_API_KEY
from jinja2 import Template
import os
import time

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_resolution(query: str):
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return {
            "resolution": "No similar incidents found.",
            "render_text": "❌ No matching past incidents were found in the system."
        }

    doc, score = results[0]
    metadata = doc.metadata or {}
    content = doc.page_content or ""

    # ✅ Limit content length to prevent Groq overload
    short_content = content[:2000]

    # Extract metadata fields safely
    ticket_id = metadata.get("ticket_id", "N/A")
    assignee = metadata.get("assignee", "someone")
    severity = metadata.get("severity", "unknown")
    comments = metadata.get("comments", [])[:5]  # Top 5 comments max

    # ✅ Format comments as bullet list
    if comments:
        comments_html = "<ul>\n"
        for comment in comments:
            author = comment.get("author", "unknown")
            body = comment.get("body", "No comment")
            comments_html += f"<li><strong>{author}</strong>: {body}</li>\n"
        comments_html += "</ul>"
    else:
        comments_html = "<p>No comments found for this incident.</p>"

    # ✅ Build LLM prompt
    prompt = (
        f"You are analyzing a new reported incident:\n"
        f"{query}\n\n"
        f"We found a similar past incident with the following details:\n"
        f"{short_content}\n\n"
        f"Summarize the resolution clearly. "
        f"Include who resolved it and what the fix was. Also note the severity level."
    )

    llm = ChatGroq(model_name=MODEL_NAME, temperature=0)

    # ✅ LLM call with retry and failover
    resolution = "Unable to generate resolution."
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            resolution = response.content.strip()
            break
        except Exception as e:
            print(f"[Groq] Attempt {attempt + 1} failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    else:
        resolution = "⚠️ Resolution could not be generated due to system error."

    # ✅ Final UI render
    render_template = Template("""
        ✅ This issue was previously resolved in <strong>{{ ticket_id }}</strong> by {{ resolution }}.
        It was handled by <strong>{{ assignee }}</strong> and categorized as a <strong>{{ severity }}</strong> incident.
        <br /><br />
        🗒️ <strong>Resolution Comments:</strong><br />
        {{ comments_html | safe }}
    """)

    render_text = render_template.render(
        resolution=resolution,
        ticket_id=ticket_id,
        assignee=assignee,
        severity=severity,
        comments_html=comments_html
    )

    return {
        "resolution": resolution,
        "ticket_id": ticket_id,
        "assignee": assignee,
        "severity": severity,
        "comments": comments,
        "render_text": render_text
    }
