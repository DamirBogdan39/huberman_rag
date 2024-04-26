from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from src.bge_embeddigns import bge_embeddings

llm = OpenAI()

HYDE_PROMPT = """Answer the question is the style of doctor Andrew Huberman.\nQuestion: {question}\nAnswer:"""

hyde_prompt = PromptTemplate.from_template(input_variable=["question"], template=HYDE_PROMPT)

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(llm=llm,
                                                        base_embeddings=bge_embeddings,
                                                        custom_prompt=hyde_prompt)