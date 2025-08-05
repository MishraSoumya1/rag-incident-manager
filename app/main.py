from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import IncidentQuery
from app.qa_service import get_resolution

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/query")
def query_incident(payload: IncidentQuery):
    resolution = get_resolution(payload.query)
    return {"resolution": resolution}
