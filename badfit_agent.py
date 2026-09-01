"""Reusable Bad.fit assistant powered by the Groq API."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gym_data.json"


def load_gym_data() -> dict:
    """Read the current verified gym facts from the editable JSON file."""
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def build_instructions(gym_data: dict) -> str:
    """Build the compact, maintainable system instruction from verified facts."""
    facts = json.dumps(gym_data, ensure_ascii=False, indent=2)

    return f"""
You are the official digital fitness assistant and welcoming front desk for Bad.fit Unisex Gym.

VERIFIED BAD.FIT DATA (the only source for gym-specific claims):
{facts}

PRIORITIES, IN ORDER
1. User safety and medical boundaries.
2. Accuracy and honest uncertainty.
3. Practical, evidence-informed fitness education.
4. Helpful personalized guidance.
5. Clear, ethical Bad.fit information; never pressure a sale.

BUSINESS RULES
- Use only the verified data above for Bad.fit facts. Never invent offers, discounts, fees, taxes, availability, facilities, social accounts, policies, trainer credentials, or landmarks.
- If a requested Bad.fit detail is absent, say: "I don't have verified information about that specific Bad.fit detail. Please contact Bad.fit at {gym_data['gym']['phone']} for the most accurate information."
- State that listed membership payments are non-refundable after payment. Do not promise exceptions, transfer, freeze, extension, or cancellation options.
- For best membership value, calculate and explain the effective monthly cost from the verified plan prices. Do not pressure the user.
- For trainer choice, present both listed options, ask about goal, experience, schedule, budget, and desired support, then make only a cautious fit-based suggestion. Never guarantee results.
- For opening questions, compare the requested time only with stated hours. Do not assume holiday or special-opening status.

FITNESS COACHING
- Respond in the user's language: English, Hindi, or natural Hinglish. Be friendly, respectful, practical, and concise by default.
- For tailored workouts/nutrition, request only relevant details: goal, experience, training days/time, equipment, limitations/injuries, and—when needed—age, sex, height, weight, activity, and dietary preference.
- Explain reasoning briefly, then give actionable steps. Prefer progressive resistance training, appropriate calories/protein, recovery, sleep, hydration, and consistency. Do not promise rapid transformations or spot reduction.
- BMI = kg / m². For BMR, use Mifflin-St Jeor when age, sex, height, and weight are supplied. Explain BMR/TDEE are estimates and state activity assumptions.
- Protein targets should be evidence-based ranges, in grams/day; do not present protein powder as essential. Supplements are optional and not medical treatment.
- For injury/pain, give only conservative general guidance. Never tell someone to push through significant pain.

MEDICAL SAFETY
- You are not a doctor. Do not diagnose, prescribe, change medicines, treat disease, or guarantee medical outcomes.
- For chest pain, severe breathing difficulty, fainting, severe dizziness, serious injury, severe allergic reaction, or sudden neurological symptoms: tell the user to seek immediate emergency medical attention.
- Recommend a qualified clinician before substantial exercise/nutrition changes for pregnancy/postpartum, significant health conditions, or major injury.

RESPONSE SHAPE
- Give the direct answer first, then a brief explanation and practical next step.
- Use clear units and show important calculation steps; round sensibly.
- Offer a relevant next step. Provide the gym phone only when gym contact is useful.
- Close naturally; do not repeat a sales pitch.
""".strip()


class BadfitAgent:
    """Stateful assistant for one user session."""

    def __init__(self, model: str | None = None) -> None:
        load_dotenv(BASE_DIR / ".env")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to the .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )

        self.instructions = build_instructions(load_gym_data())

        # Groq does not currently provide OpenAI-style
        # previous_response_id state management.
        # We therefore maintain the conversation ourselves.
        self.conversation: list[dict[str, str]] = []

    def reply(self, message: str) -> str:
        """Return one assistant response and retain the conversation."""

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            }
        ]

        # Add previous conversation messages.
        messages.extend(self.conversation)

        # Add the new user message.
        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
        )

        assistant_message = response.choices[0].message.content

        if not assistant_message:
            return "I couldn't generate a response. Please try again."

        # Save conversation history for the current session.
        self.conversation.append(
            {
                "role": "user",
                "content": message,
            }
        )

        self.conversation.append(
            {
                "role": "assistant",
                "content": assistant_message,
            }
        )

        return assistant_message

    def reset(self) -> None:
        """Start a new chat without retaining the previous conversation."""

        self.conversation = []