from langchain_classic.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

from config.models import models
from src.bge_embeddigns import bge_embeddings

llm = ChatAnthropic(model=models.sonnet_4_6)

HYDE_PROMPT = """Answer the question is the style of doctor Andrew Huberman.\nQuestion: {question}\nAnswer:"""

hyde_prompt = PromptTemplate.from_template(
    input_variable=["question"], template=HYDE_PROMPT
)

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm, base_embeddings=bge_embeddings, custom_prompt=hyde_prompt
)
