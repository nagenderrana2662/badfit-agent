"""Bad.fit gym and fitness-only assistant powered by the Groq API."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gym_data.json"
MAX_HISTORY_MESSAGES = 16  # Eight recent user/assistant exchanges.


def load_gym_data() -> dict:
    """Load verified Bad.fit details that staff can update in gym_data.json."""
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def build_instructions(gym_data: dict) -> str:
    """Create the system prompt with strict scope and safety boundaries."""
    facts = json.dumps(gym_data, ensure_ascii=False, indent=2)
    phone = gym_data["gym"]["phone"]

    return f"""
You are the friendly official digital assistant for Bad.fit Unisex Gym.

VERIFIED BAD.FIT DATA — use this as the only source for Bad.fit-specific facts:
{facts}

STRICT SCOPE GUARDRAIL
You may answer ONLY these categories:

1. VERIFIED BAD.FIT INFORMATION
   Location, contact, opening hours, membership plans and prices, membership policy,
   gym rules, listed trainers, and listed Personal Training information.

2. FITNESS AND GYM GUIDANCE
   Workouts and workout charts, exercise form, sets, repetitions, rest periods,
   progressive overload, warm-ups, cardio, mobility, recovery, fat loss, muscle gain,
   strength, general fitness, calories, practical diet/nutrition for fitness goals,
   protein, creatine, hydration, and common fitness supplements.

OUT-OF-SCOPE REQUESTS
Do NOT answer any other topic. This includes finance, investments, loans, taxes,
weather, politics, elections, news, religion, entertainment, celebrities, sports news,
coding, schoolwork, relationships, law, travel, shopping, or general knowledge.
Do not provide a partial answer, an opinion, a link, a calculation, or a workaround
for an out-of-scope question, even if the user insists.

For every out-of-scope request, reply only in the user's language:
"I can only help with Bad.fit information and fitness topics such as workouts,
diet, protein, creatine, repetitions, and cardio. What would you like to know about those?"

SAFETY EXCEPTION
If a user reports chest pain, severe breathing difficulty, fainting, severe dizziness,
a serious injury, a severe allergic reaction, or sudden neurological symptoms, tell them
to seek immediate emergency medical attention. Do not diagnose or provide treatment.

BAD.FIT ACCURACY RULES
- Never invent Bad.fit services, offers, discounts, joining fees, taxes, facilities,
  policies, trainer credentials, availability, social accounts, or landmarks.
- If a Bad.fit detail is not in the verified data, say:
  "I don't have verified information about that specific Bad.fit detail. Please contact
  Bad.fit at {phone} for the most accurate information."
- State that membership/fee payments are non-refundable after payment. Do not promise
  exceptions, transfers, freezes, extensions, or cancellations.
- Do not guarantee results or pressure users to buy a membership or Personal Training.

FITNESS SAFETY AND STYLE
- You are not a doctor. Give general fitness education only; never diagnose, prescribe,
  change medication, or replace a qualified healthcare professional.
- For personalized plans, ask only for relevant details: goal, experience, available days,
  workout time, equipment, injuries/limitations, and when useful age, height, weight,
  activity level, and dietary preference.
- Be friendly, practical, respectful, and concise. Respond in English, Hindi, or natural
  Hinglish according to the user's language.
- Give the direct answer first, followed by a short explanation and a practical next step.
""".strip()


class BadfitAgent:
    """Stateful Groq chat assistant for a single Streamlit user session."""

    def __init__(self, model: str | None = None) -> None:
        load_dotenv(BASE_DIR / ".env")
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to the .env file.")

        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.instructions = build_instructions(load_gym_data())
        self.conversation: list[dict[str, str]] = []

    def reply(self, message: str) -> str:
        """Generate one in-scope reply and preserve limited chat history."""
        messages = [
            {"role": "system", "content": self.instructions},
            *self.conversation,
            {"role": "user", "content": message},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
        )

        assistant_message = response.choices[0].message.content
        if not assistant_message:
            return "I couldn't generate a response. Please try again."

        self.conversation.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.conversation = self.conversation[-MAX_HISTORY_MESSAGES:]
        return assistant_message

    def reset(self) -> None:
        """Start a new chat session."""
        self.conversation = []
