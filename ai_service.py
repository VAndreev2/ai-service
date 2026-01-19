from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
vector_store = Chroma(
    collection_name="stepik_courses",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

app = FastAPI()

class GeneratePayload(BaseModel):
    messages: list
    schema: dict
    temperature: float
    model: str

class RagRequest(BaseModel):
    profile: str
    k: int = 3
    
@app.post("/generate")
def generate(payload: GeneratePayload):
    response = client.chat.completions.create(
        model=payload.model,
        messages=payload.messages,
        temperature=payload.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": payload.schema,
        },
    )
    return {
        "content": response.choices[0].message.content
    }

@app.post("/rag")
def rag_endpoint(req: RagRequest):
    docs = vector_store.similarity_search(req.profile, k=req.k)

    docs_content = json.dumps(
        [
            {
                "title": d.metadata.get("title", ""),
                "summary": d.metadata.get("summary", ""),
                "link": d.metadata.get("link", ""),
                "level": d.metadata.get("level", "basic"),
            }
            for d in docs
        ],
        ensure_ascii=False
    )

    prompt = f"""
    Доступные курсы:
    {docs_content}

    Подбери {req.k} наиболее подходящих курсов под профиль:
    {req.profile}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return {"content": response.choices[0].message.content}
