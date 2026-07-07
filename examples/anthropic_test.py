import os

import anthropic
from dotenv import load_dotenv

from config.models import models

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model=models.claude,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What are the main health benefits of morning sunlight exposure?"}
    ],
)

print(message.content[0].text)
