import litellm
import os

if os.getenv("OPENAI_API_KEY"):
    litellm.openai_key = os.getenv("OPENAI_API_KEY")

if (litellm.openai_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected vocareum API key. Using Vocareum OpenAI API base URL.")