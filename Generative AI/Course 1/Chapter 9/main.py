# Student Task: Set up the OpenAI API key and base URL from environment variables

import litellm
import os
import numpy as np

if os.getenv("OPENAI_API_KEY"):
    litellm.openai_key = os.getenv("OPENAI_API_KEY")

if (litellm.openai_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Using Vocareum OpenAI API base URL")