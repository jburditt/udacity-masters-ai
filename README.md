# udacity-masters-ai
Udacity Masters in AI

## Setup
- Run `python -m venv .venv`
- Run `.venv\Scripts\Activate`
- Run `python -m pip install --upgrade pip`
- Run `python -m pip install requests`

## Run
- Run `python test.py`

### OpenAI Python Package Version 0
import openai
openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = "voc-00000000000000000000000000000000abcd.12345678"

### OpenAI Python Package Version 1
from openai import OpenAI
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key = "voc-00000000000000000000000000000000abcd.12345678"
)

## REST API Calls
If using curl or another client to access the OpenAI API, build the URLs using https://openai.vocareum.com/v1 in place of https://api.openai.com/v1

