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

TASK1_DESCRIPTORS = """
IELTS WRITING TASK 1 - OFFICIAL BAND DESCRIPTORS (Updated May 2023)
Each criterion is scored as a WHOLE NUMBER only (1-9). No decimals. No .5.

TASK ACHIEVEMENT (TA):
9: All requirements fully and appropriately satisfied. Extremely rare lapses.
8: Covers all requirements appropriately, relevantly, sufficiently. Key features skilfully selected, presented, highlighted, illustrated. Occasional omissions.
7: Covers requirements. Content relevant and accurate with few omissions. Key features covered and clearly highlighted. Clear overview, data appropriately categorised, main trends identified.
6: Focuses on requirements. Key features covered and adequately highlighted. Overview attempted. Some irrelevant/inaccurate info may occur. Some details missing or excessive.
5: Generally addresses requirements. Key features not adequately covered. Recounting mainly mechanical. May be no data to support description. Tendency to focus on details without bigger picture.
4: Attempts to address task. Few key features selected. Key features may be irrelevant, repetitive, inaccurate. Format may be inappropriate.
3: Does not address requirements (possibly misunderstood data/diagram). Key features largely irrelevant. Limited information, used repetitively.
2: Content barely relates to task.
1: Content wholly unrelated to task (or 20 words or fewer).

COHERENCE & COHESION (CC):
9: Message followed effortlessly. Cohesion rarely attracts attention. Minimal lapses. Paragraphing skilfully managed.
8: Message followed with ease. Ideas logically sequenced, cohesion well managed. Occasional lapses. Paragraphing used sufficiently and appropriately.
7: Ideas logically organised, clear progression. Few lapses. Cohesive devices used flexibly but with some inaccuracies or over/under use.
6: Generally arranged coherently, clear overall progression. Cohesive devices used to some good effect but cohesion may be faulty or mechanical. Reference/substitution may lack flexibility.
5: Organisation evident but not wholly logical, may lack overall progression. Relationship of ideas can be followed but sentences not fluently linked. Limited/overuse of cohesive devices.
4: Ideas evident but not arranged coherently, no clear progression. Relationships unclear/inadequately marked. Inaccurate use or lack of substitution/referencing.
3: No apparent logical organisation. Minimal use of sequencers/cohesive devices. Difficulty identifying referencing.
2: Little relevant message. Little evidence of control of organisational features.
1: Writing fails to communicate any message.

LEXICAL RESOURCE (LR):
9: Full flexibility and precise use evident. Wide range used accurately and appropriately. Very natural and sophisticated control. Minor errors extremely rare.
8: Wide resource fluently and flexibly used to convey precise meanings. Skilful use of uncommon/idiomatic items. Occasional errors in spelling/word formation with minimal impact.
7: Sufficient for some flexibility and precision. Some ability to use less common/idiomatic items. Awareness of style and collocation, though inappropriacies occur. Few errors in spelling/word formation.
6: Generally adequate and appropriate. Meaning generally clear despite restricted range or lack of precision. Some errors in spelling/word formation but do not impede communication.
5: Limited but minimally adequate. Simple vocabulary used accurately but range doesn't permit much variation. Frequent lapses in appropriacy. Errors in spelling/word formation may cause difficulty.
4: Limited and inadequate for task. Basic vocabulary, may be used repetitively. Inappropriate word choice/errors in word formation may impede meaning.
3: Inadequate. Possible over-dependence on memorised language. Control of word choice/spelling very limited, errors predominate and may severely impede meaning.
2: Extremely limited. Few recognisable strings apart from memorised phrases. No apparent control of word formation/spelling.
1: No resource apparent except a few isolated words.

GRAMMATICAL RANGE & ACCURACY (GRA):
9: Wide range of structures used with full flexibility and control. Punctuation and grammar used appropriately throughout. Minor errors extremely rare.
8: Wide range of structures flexibly and accurately used. Majority of sentences error-free, punctuation well managed. Occasional non-systematic errors with minimal impact.
7: Variety of complex structures used with some flexibility and accuracy. Grammar and punctuation generally well controlled. Error-free sentences frequent. Few errors persist but don't impede communication.
6: Mix of simple and complex sentence forms but flexibility limited. More complex structures not as accurate as simple. Errors in grammar and punctuation occur but rarely impede communication.
5: Range of structures limited and rather repetitive. Complex sentences attempted but tend to be faulty. Grammatical errors may be frequent and cause difficulty. Punctuation may be faulty.
4: Very limited range of structures. Subordinate clauses rare, simple sentences predominate. Grammatical errors frequent and may impede meaning. Punctuation often faulty.
3: Sentence forms attempted but errors in grammar and punctuation predominate. Prevents most meaning from coming through. Length may be insufficient.
2: Little or no evidence of sentence forms except in memorised phrases.
1: No rateable language evident.

OVERALL BAND CALCULATION for Task 1:
Average = (TA + CC + LR + GRA) / 4
Round to nearest 0.5: if average ends in .25 round up to .5, if ends in .75 round up to next whole number.
Example: (6+6+7+6)/4 = 6.25 → Overall Band 6.5
"""

TASK2_DESCRIPTORS = """
IELTS WRITING TASK 2 - OFFICIAL BAND DESCRIPTORS (Updated May 2023)
Each criterion is scored as a WHOLE NUMBER only (1-9). No decimals. No .5.

TASK RESPONSE (TR):
9: Prompt appropriately addressed and explored in depth. Clear and fully developed position which directly answers question. Ideas relevant, fully extended and well supported.
8: Prompt appropriately and sufficiently addressed. Clear and well-developed position. Ideas relevant, well extended and supported. Occasional omissions.
7: Main parts of prompt appropriately addressed. Clear and developed position. Main ideas extended and supported but may over-generalise or lack focus/precision.
6: Main parts addressed (some more fully than others). Position directly relevant to prompt but conclusions may be unclear/unjustified. Main ideas relevant but some insufficiently developed.
5: Main parts incompletely addressed. Format may be inappropriate. Position expressed but development not always clear. Some main ideas put forward but limited and not sufficiently developed.
4: Prompt tackled minimally or tangentially. Position discernible but hard to find. Main ideas difficult to identify. Large parts may be repetitive.
3: No part adequately addressed or prompt misunderstood. No relevant position. Few ideas, may be irrelevant or insufficiently developed.
2: Content barely related to prompt. No position identifiable.
1: Content wholly unrelated to prompt (or 20 words or fewer).

COHERENCE & COHESION (CC):
9: Message followed effortlessly. Cohesion rarely attracts attention. Minimal lapses. Paragraphing skilfully managed.
8: Message followed with ease. Ideas logically sequenced, cohesion well managed. Occasional lapses. Paragraphing used sufficiently and appropriately.
7: Ideas logically organised, clear progression throughout. Few minor lapses. Cohesive devices used flexibly but with some inaccuracies or over/under use. Paragraphing generally effective.
6: Generally arranged coherently, clear overall progression. Cohesive devices used to some good effect but cohesion may be faulty or mechanical. Paragraphing may not always be logical.
5: Organisation evident but not wholly logical, may lack overall progression. Relationship of ideas can be followed but sentences not fluently linked. Paragraphing may be inadequate or missing.
4: Ideas evident but not arranged coherently, no clear progression. Relationships unclear. May be no paragraphing or no clear main topic within paragraphs.
3: No apparent logical organisation. Minimal use of sequencers/cohesive devices. Difficulty identifying referencing. Any paragraphing attempts are unhelpful.
2: Little relevant message. Little evidence of control of organisational features.
1: Writing fails to communicate any message.

LEXICAL RESOURCE (LR):
9: Full flexibility and precise use widely evident. Wide range used accurately and appropriately. Very natural and sophisticated control. Minor errors extremely rare.
8: Wide resource fluently and flexibly used to convey precise meanings. Skilful use of uncommon/idiomatic items. Occasional errors with minimal impact.
7: Sufficient for some flexibility and precision. Some ability to use less common/idiomatic items. Awareness of style and collocation, though inappropriacies occur. Few errors.
6: Generally adequate and appropriate. Meaning generally clear despite restricted range or lack of precision. Some errors but do not impede communication.
5: Limited but minimally adequate. Simple vocabulary used accurately but range doesn't permit much variation. Frequent lapses in appropriacy. Errors may cause difficulty.
4: Limited and inadequate for task. Basic vocabulary, may be used repetitively. Errors in word formation/spelling may impede meaning.
3: Inadequate. Possible over-dependence on memorised language. Control very limited, errors predominate and may severely impede meaning.
2: Extremely limited. Few recognisable strings. No apparent control.
1: No resource apparent.

GRAMMATICAL RANGE & ACCURACY (GRA):
9: Wide range of structures used with full flexibility and control. Grammar and punctuation used appropriately throughout. Minor errors extremely rare.
8: Wide range flexibly and accurately used. Majority of sentences error-free. Occasional non-systematic errors with minimal impact.
7: Variety of complex structures with some flexibility and accuracy. Grammar and punctuation generally well controlled. Error-free sentences frequent. Few errors persist.
6: Mix of simple and complex forms but flexibility limited. More complex structures not as accurate. Errors in grammar and punctuation occur but rarely impede communication.
5: Range limited and rather repetitive. Complex sentences tend to be faulty. Grammatical errors may be frequent and cause difficulty. Punctuation may be faulty.
4: Very limited range. Simple sentences predominate. Grammatical errors frequent and may impede meaning. Punctuation often faulty.
3: Sentence forms attempted but errors predominate. Prevents most meaning from coming through.
2: Little or no evidence of sentence forms.
1: No rateable language evident.

OVERALL BAND CALCULATION for Task 2:
Average = (TR + CC + LR + GRA) / 4
Round to nearest 0.5: if average ends in .25 round up to .5, if ends in .75 round up to next whole number.
Example: (6+7+6+6)/4 = 6.25 → Overall Band 6.5
Example: (7+7+7+6)/4 = 6.75 → Overall Band 7.0
"""

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
        criterion1_name = "Task Achievement (TA)"
        descriptors = TASK1_DESCRIPTORS
    else:
        criterion1_name = "Task Response (TR)"
        descriptors = TASK2_DESCRIPTORS

    system_prompt = (
        "You are a certified IELTS examiner with 10+ years of experience. "
        "You must grade strictly and accurately following the official IELTS Band Descriptors below.\n\n"
        + descriptors +
        "\n\nCRITICAL RULES:\n"
        "1. Each criterion score MUST be a WHOLE NUMBER (1-9). NEVER use decimals like 6.5 for individual criteria.\n"
        "2. Only the overall_band can be x.0 or x.5 (calculated by averaging the 4 criteria and rounding to nearest 0.5).\n"
        "3. Read the band descriptors carefully for EACH criterion before scoring.\n"
        "4. Be strict and accurate — do not inflate scores.\n"
        "5. Respond ONLY in this exact JSON format, no extra text, no markdown:\n"
        "{\n"
        f'  "criterion1_name": "{criterion1_name}",\n'
        '  "criterion1_score": 6,\n'
        '  "criterion1_feedback": "Specific feedback referencing the band descriptors...",\n'
        '  "criterion2_name": "Coherence & Cohesion",\n'
        '  "criterion2_score": 6,\n'
        '  "criterion2_feedback": "Specific feedback referencing the band descriptors...",\n'
        '  "criterion3_name": "Lexical Resource",\n'
        '  "criterion3_score": 7,\n'
        '  "criterion3_feedback": "Specific feedback referencing the band descriptors...",\n'
        '  "criterion4_name": "Grammatical Range & Accuracy",\n'
        '  "criterion4_score": 6,\n'
        '  "criterion4_feedback": "Specific feedback referencing the band descriptors...",\n'
        '  "overall_band": 6.5,\n'
        '  "overall_feedback": "Comprehensive examiner feedback on the essay overall..."\n'
        "}"
    )

    user_content = f"EXAM PROMPT:\n{req.prompt}\n\nSTUDENT ESSAY:\n{req.essay}"
    user_message = f"Grade this IELTS Writing {req.task_type} using the official band descriptors provided:\n\n{user_content}\n/no_think"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=messages,
            max_tokens=2000,
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
        result = json.loads(raw.strip())

        # Force integer scores for each criterion
        for key in ["criterion1_score", "criterion2_score", "criterion3_score", "criterion4_score"]:
            if key in result:
                result[key] = int(round(result[key]))

        # Recalculate overall band correctly
        scores = [result["criterion1_score"], result["criterion2_score"],
                  result["criterion3_score"], result["criterion4_score"]]
        avg = sum(scores) / 4
        # Round to nearest 0.5
        result["overall_band"] = round(avg * 2) / 2

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
