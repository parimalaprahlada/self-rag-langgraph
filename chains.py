import os
from dotenv import load_dotenv
from typing import Literal
from pydantic.v1 import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import (
    rag_prompt, db_query_rewrite_prompt, hallucination_prompt, 
    answer_prompt, query_feedback_prompt, generation_feedback_prompt, give_up_prompt, grade_doc_prompt,
    knowledge_extraction_prompt, router_prompt, websearch_query_rewrite_prompt, simple_question_prompt
)
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

base_url = os.environ["base_url"]
api_key = os.environ["api_key"]
tavily_key = os.environ["TAVILY_API_KEY"]

web_search_tool = TavilySearchResults(k=3)

class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in facts, 'yes' or 'no' ")

class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer addresses the question, 'yes' or 'no'")

class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Document is relevant to the question, 'yes' or 'no' ")

class RouteQuery(BaseModel):
    route: Literal["vectorstore","websearch","QA_LM"] = Field(description="Given a user question choose to route it to web search (websearch), a vectorstore (vectorstore), or a QA language model (QA_LM).")

llm_engine = ChatOpenAI(model="groq/openai/gpt-oss-20b", base_url=base_url, api_key=api_key)

rag_chain = rag_prompt|llm_engine|StrOutputParser()

db_query_rewriter = db_query_rewrite_prompt|llm_engine|StrOutputParser()

hallucination_grader = (
    hallucination_prompt | llm_engine.with_structured_output(GradeHallucination)
)

answer_grader = (
    answer_prompt | llm_engine.with_structured_output(GradeAnswer)
)

query_feedback_chain = (
    query_feedback_prompt|llm_engine|StrOutputParser()
)

generation_feedback_chain = (
    generation_feedback_prompt|llm_engine|StrOutputParser()
)

give_up_chain = ( give_up_prompt|llm_engine|StrOutputParser() )

retrieval_grader = (grade_doc_prompt|llm_engine.with_structured_output(GradeDocuments))

knowledge_extractor= (knowledge_extraction_prompt|llm_engine|StrOutputParser())

question_router = (router_prompt|llm_engine.with_structured_output(RouteQuery))

websearch_query_rewriter = (websearch_query_rewrite_prompt|llm_engine|StrOutputParser())

simple_question_chain = (simple_question_prompt|llm_engine|StrOutputParser())