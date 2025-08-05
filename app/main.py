from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.models import IncidentQuery
from app.qa_service import get_resolution
from app.config import CHROMA_DIR

app = FastAPI()

# Enable CORS for all domains and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once

@app.post("/query")
def query_incident(payload: IncidentQuery):
    resolution = get_resolution(payload.query)
    return {"resolution": resolution}

@app.post("/ingest")
async def ingest_incidents(
    authorization: Optional[str] = Header(None),
    file: Optional[UploadFile] = File(None)
):
    # Authorization check
    if authorization != "Bearer mysecrettoken":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load incidents
    try:
        if file:
            content = await file.read()
            incidents = json.loads(content)
        else:
            with open("mock_incidents.json", "r") as f:
                incidents = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read incident data: {str(e)}")

    # Convert incidents to Documents
    docs = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    for incident in incidents:
        comment_text = "\n".join(
            [f"{c['author']}: {c['body']}" for c in incident.get("comments", [])]
        )

        content = (
            f"Title: {incident['title']}\n"
            f"Description: {incident['description']}\n"
            f"Comments:\n{comment_text}\n"
            f"Fix: {incident['fix_summary']}\n"
        )

        # 🛠️ FIXED: Flatten comments into a string (as metadata must be primitive types)
        metadata = {
            "ticket_id": incident["ticket_id"],
            "assignee": incident["assignee"],
            "severity": incident["severity"],
            "comments": comment_text
        }

        docs.append(Document(page_content=content, metadata=metadata))

    # Store in vector DB
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    return JSONResponse(content={"status": "success", "count": len(docs)}, status_code=200)
