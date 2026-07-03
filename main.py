import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_TOKEN = os.environ.get("HF_TOKEN", "")

class GradeRequest(BaseModel):
    essay: str
    prompt: str
    task_type: str
    image_base64: Optional[str] = None

@app.get("/")
def root():
    return {"status": "LND Academy IELTS Grader API is running"}

@app.post("/grade")
def grade_essay(req: GradeRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    if len(req.essay.strip()) < 50:
        raise HTTPException(status_code=400, detail="Essay is too short")

    client = InferenceClient(provider="nscale", api_key=HF_TOKEN)

    if req.task_type == "Task 1":
        criterion1 = "Task Achievement (TA)"
        task_instruction = (
            "This is an IELTS Writing Task 1. The student was given a chart/graph/diagram "
            "and must describe its key features accurately. Check if the essay correctly "
            "describes the data and addresses the prompt."
        )
    else:
        criterion1 = "Task Response (TR)"
        task_instruction = (
            "This is an IELTS Writing Task 2 argumentative essay. "
            "Check if the essay fully addresses all parts of the prompt."
        )

    system_prompt = (
        "You are a certified IELTS examiner with 10+ years of experience. "
        + task_instruction +
        " Grade strictly following official IELTS Band Descriptors (0-9 scale, .5 increments allowed). "
        "Respond ONLY in this exact JSON format, no extra text, no markdown:\n"
        "{\n"
        f'  "criterion1_name": "{criterion1}",\n'
        '  "criterion1_score": 7.0,\n'
        '  "criterion1_feedback": "...",\n'
        '  "criterion2_name": "Coherence & Cohesion",\n'
        '  "criterion2_score": 6.5,\n'
        '  "criterion2_feedback": "...",\n'
        '  "criterion3_name": "Lexical Resource",\n'
        '  "criterion3_score": 7.0,\n'
        '  "criterion3_feedback": "...",\n'
        '  "criterion4_name": "Grammatical Range & Accuracy",\n'
        '  "criterion4_score": 6.5,\n'
        '  "criterion4_feedback": "...",\n'
        '  "overall_band": 7.0,\n'
        '  "overall_feedback": "..."\n'
        "}"
    )

    user_content = f"EXAM PROMPT:\n{req.prompt}\n\nSTUDENT ESSAY:\n{req.essay}"
    user_message = f"Grade this IELTS Writing {req.task_type}:\n\n{user_content}\n/no_think"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=messages,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
