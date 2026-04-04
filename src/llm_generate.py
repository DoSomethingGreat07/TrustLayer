# llm_generate.py

from openai import OpenAI
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load .env file
load_dotenv()

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables"
    )

# Initialize client
client = OpenAI(api_key=api_key)




TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_grounded_answer",
            "description": "Generate a grounded answer strictly from the provided evidence context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Concise final answer to the user's question."
                    },
                    "justification": {
                        "type": "string",
                        "description": "Short explanation grounded only in the provided evidence."
                    }
                },
                "required": ["answer", "justification"]
            }
        }
    }
]


def llm_generate_fn(prompt: str) -> str:
    """
    Plug directly into your current corrective_rag.py
    because your pipeline currently calls:

        answer = llm_generate_fn(prompt)

    So this returns a single string combining answer + justification.
    """

    system_prompt = """
You are a grounded research assistant.

Rules:
- Use ONLY the provided evidence.
- Do NOT use outside knowledge.
- If the evidence is insufficient, answer exactly:
  Insufficient evidence
- Keep the answer concise and factual.
- The justification must be short and evidence-based.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {
                "name": "generate_grounded_answer"
            }
        },
    )

    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    answer = arguments.get("answer", "Insufficient evidence").strip()
    justification = arguments.get("justification", "").strip()

    if answer.lower() == "insufficient evidence":
        return "Insufficient evidence"

    if justification:
        return f"{answer}\n\nJustification: {justification}"

    return answer