from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
app = FastAPI()

class GeneratePayload(BaseModel):
    messages: list
    schema: dict
    temperature: float
    model: str

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
