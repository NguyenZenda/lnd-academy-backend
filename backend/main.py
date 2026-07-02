from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

class GradeRequest(BaseModel):
    essay: str
    task_type: str  # "Task 1" or "Task 2"

@app.get("/")
def root():
    return {"status": "LND Academy IELTS Grader API is running"}

@app.post("/grade")
def grade_essay(req: GradeRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")

    if len(req.essay.strip()) < 50:
        raise HTTPException(status_code=400, detail="Essay is too short")

    client = InferenceClient(provider="groq", api_key=HF_TOKEN)

    if req.task_type == "Task 1":
        task_instruction = "This is an IELTS Writing Task 1 (describing a graph, chart, diagram, or map)."
        criteria = "Task Achievement (TA)"
    else:
        task_instruction = "This is an IELTS Writing Task 2 (argumentative/discussion essay)."
        criteria = "Task Response (TR)"

    system_prompt = (
        "You are a certified IELTS examiner with 10+ years of experience. "
        f"{task_instruction} "
        "Grade strictly following official IELTS Band Descriptors (0-9 scale, including .5 increments). "
        "Respond ONLY in this exact JSON format, no extra text:\n"
        "{\n"
        f'  "criterion1_name": "{criteria}",\n'
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

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Grade this IELTS Writing {req.task_type} essay:\n\n{req.essay}"}
            ],
            max_tokens=1500,
        )

        import json
        raw = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
