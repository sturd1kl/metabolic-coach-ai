from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# Securely load API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UserInput(BaseModel):
    age: int
    sex: str
    height: float
    weight: float
    goal_weight: float
    activity_level: str
    glp1_user: bool
    medication: str = ""
    symptoms: str = ""

SYSTEM_PROMPT = """
You are a clinical metabolic coach designed by a physician assistant.

You provide:
- Conservative, evidence-based weight loss guidance
- Structured plans (not conversational output)

Two user types:
1. GLP-1 users → focus on protein, prevent under-eating, manage side effects
2. Non-GLP-1 users → focus on satiety, adherence, sustainable deficit

Rules:
- Target weight loss: 0.5–1% body weight per week
- Avoid aggressive calorie restriction
- Prioritize protein intake
- Flag safety concerns if intake too low or symptoms severe

Output format:

Calorie Target:
Protein Target:

3 Key Actions:
- 
- 
- 

GLP-1 / Metabolic Optimization:
-

Safety Note:
-
"""

@app.get("/")
def read_root():
    return {"message": "Metabolic Coach AI is running"}

@app.post("/generate-plan")
def generate_plan(user: UserInput):

    user_context = f"""
    Age: {user.age}
    Sex: {user.sex}
    Height: {user.height}
    Weight: {user.weight}
    Goal Weight: {user.goal_weight}
    Activity Level: {user.activity_level}
    GLP-1 User: {user.glp1_user}
    Medication: {user.medication}
    Symptoms: {user.symptoms}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_context}
        ],
        temperature=0.4
    )

    return {
        "plan": response.choices[0].message.content
    }
