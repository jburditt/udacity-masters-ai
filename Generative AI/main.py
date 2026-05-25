import litellm
import os

if os.getenv("OPENAI_API_KEY"):
    litellm.openai_key = os.getenv("OPENAI_API_KEY")

if (litellm.openai_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected vocareum API key. Using Vocareum OpenAI API base URL.")
    
SYSTEM_PROMPT = """You are a helpful coding assistant. Return only the Python code, with no explanation or preamble."""
USER_PROMPT = """Write a Python function that implements the factorial algorithm using recursion.
The name of the function should be `factorial`.
The function should take a single non-negative integer as input.
Include a docstring explaining what the function does."""

from litellm import completion

response = completion(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print("Generated Code:\n")
print(response.choices[0].message.content)