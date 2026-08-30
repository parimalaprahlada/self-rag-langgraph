from typing import List, Literal, Optional, TypedDict
from pydantic.v1 import BaseModel
from data_index import retriever, REPO_IDENTIFIER
from chains import (rag_chain, db_query_rewriter, hallucination_grader, answer_grader, 
     generation_feedback_chain, query_feedback_chain, give_up_chain, knowledge_extractor,
      retrieval_grader, question_router, simple_question_chain, websearch_query_rewriter, web_search_tool)
from functools import wraps


class GraphState(BaseModel):
    question: Optional[str] = None
    generation: Optional[str] = None
    documents: List[str] = []
    rewritten_question: Optional[str] = None
    query_feedback: List[str] = []
    generation_feedback: List[str] = []
    generation_number: int = 0
    search_mode: Literal["vectorstore","websearch","QA_LM"] = "QA_LM"
    retrieval_num: int = 0
    rewrite_history: List[str] = []
    repo_scoped_search: bool = False

MAX_GENERATIONS = 3
MAX_RETRIEVALS = 3

def with_state_model(model_cls):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            if isinstance(state, dict):
                state = model_cls(**state)
            return func(state)
        return wrapper
    return decorator

@with_state_model(GraphState)
def retriever_node(state: GraphState):
    print(f"DEBUG: type of state = {type(state)}")
    new_documents = retriever.invoke(state.rewritten_question)
    new_documents = [d.page_content for d in new_documents]
    state.documents.extend(new_documents)
    return {
        "documents": state.documents,
        "retrieval_num": state.retrieval_num + 1
        }


@with_state_model(GraphState)
def generation_node(state: GraphState):
    generation = rag_chain.invoke(
        {
            "context": "\n\n".join(state.documents),
            "question": state.question, 
            "feedback": "\n".join(state.generation_feedback)
        }
    )
    return {
        "generation": generation,
        "generation_number": state.generation_number + 1
        }

@with_state_model(GraphState)
def db_query_rewriting_node(state: GraphState):
    rewritten_question = db_query_rewriter.invoke(
        {
            "question": state.question, 
            "feedback": "\n".join(state.query_feedback),
            "repo_identifier": REPO_IDENTIFIER,
        }
    )
    print(f"  ├─ original question : {state.question}")
    print(f"  └─ rewritten question: {rewritten_question}")
    return {"rewritten_question": rewritten_question, "search_mode": "vectorstore", 
            "rewrite_history": state.rewrite_history + [rewritten_question],
            "repo_scoped_search": True,
            }

@with_state_model(GraphState)
def answer_evaluation_node(state: GraphState):
    hallucination_grade = hallucination_grader.invoke(
        {"documents": state.documents, "generation": state.generation}
    )

    if hallucination_grade.binary_score == "yes":
        if state.generation_number >= MAX_GENERATIONS:
            return "max_generation_reached"
        # if no hallucination, assess relevance
        answer_grade = answer_grader.invoke({
            "question": state.question, 
            "generation": state.generation
        })
        if answer_grade.binary_score == "yes":
            # no hallucination and relevant
            return "useful"
        elif state.generation_number >= MAX_GENERATIONS:
            return "max_generation_reached"
        else:
            # no hallucination but not relevant
            return "not relevant"
    elif state.generation_number >= MAX_GENERATIONS:
        return "max_generation_reached"
    else:
        # we have hallucination
        return "hallucination" 
    
@with_state_model(GraphState)
def generation_feedback_node(state: GraphState):
    feedback = generation_feedback_chain.invoke(
        {
            "question": state.question,
            "documents": "\n\n".join(state.documents), 
            "generation": state.generation
        }
    )

    feedback = 'Feedback about the answer "{}": {}'.format(state.generation, feedback)
    state.generation_feedback.append(feedback)
    return {"generation_feedbacks": state.generation_feedback}

@with_state_model(GraphState)
def query_feedback_node(state: GraphState):
    feedback = query_feedback_chain.invoke(
        {
            "question": state.question,
            "rewritten_question": state.rewritten_question, 
            "documents": "\n\n".join(state.documents),
            "generation": state.generation

        }
    )

    feedback = 'Feedback about the query "{}": {}'.format(state.rewritten_question, feedback)
    state.query_feedback.append(feedback)
    return {"query_feebacks": state.query_feedback}

@with_state_model(GraphState)
def give_up_node(state: GraphState):
    response = give_up_chain.invoke(state.question)
    return {"generation": response}

@with_state_model(GraphState)
def filter_relevant_documents_node(state: GraphState):
    grades = retrieval_grader.batch(
        [{
            "question": state.question,
            "document": doc}
            for doc in state.documents
        ]
    )

##Then we keep only the documents that were graded as relevant
    filtered_docs = [
        doc for grade, doc 
        in zip(grades, state.documents) 
        if grade.binary_score == 'yes'
    ]


# If we didn't get any relevant document, let's capture that 
    # as a feedback for the next retrieval iteration
    updated_feedback = state.query_feedback
    if not filtered_docs: 
        feedback = 'Feedback about query "{}" : did not generate any relevant documents.'.format(state.rewritten_question)
        updated_feedback = state.query_feedback + [feedback]
    else:
        updated_feedback = state.query_feedback

    return {
        "documents": filtered_docs, 
        "query_feedback": updated_feedback
    }

@with_state_model(GraphState)
def knowledge_extractor_node(state: GraphState):
    extracted = knowledge_extractor.batch([
        {"question": state.question, "document": doc } for doc in state.documents
    ])

    #we keep only non empty documents
    extracted = [doc for doc in extracted if doc]
    return {
        "documents": extracted
    }

@with_state_model(GraphState)
def router_node(state: GraphState):
    route_query = question_router.invoke(
        {
            "question": state.question, 
            "repo_identifier": REPO_IDENTIFIER,
        })
    return route_query.route

@with_state_model(GraphState)
def simple_question_node(state: GraphState):
    answer = simple_question_chain.invoke(state.question)
    return {"generation": answer, "search_mode": "QA_LM"}

@with_state_model(GraphState)
def websearch_query_rewriting_node(state: GraphState):
    rewritten_question = websearch_query_rewriter.invoke(
        {
            "question": state.question, 
            "feedback": "\n".join(state.query_feedback)
        }
    )
    if state.search_mode != "websearch":
        state.retrieval_num = 0
    return {
        "rewritten_question": rewritten_question, 
        "search_mode": "websearch",
        "retrieval_num": state.retrieval_num
    }

# @with_state_model(GraphState)
# def web_search_node(state: GraphState):
#     search_query = f"{state.rewritten_question} \"{REPO_IDENTIFIER}\""
#     new_docs = web_search_tool.invoke(
#         {"query": search_query}
#     )
#     print("DEBUG search query:", search_query)
#     for d in new_docs:
#         print("DEBUG result URL:", d.get("url", "no-url"))
#         print("DEBUG result snippet:", d["content"][:150])
#     web_results = [d["content"] for d in new_docs]
#     updated_documents = state.documents + web_results
#     # state.documents.extend(web_results)
#     return {"documents": updated_documents, "retrieval_num": state.retrieval_num + 1}


###commenting below to fix strange during runtime
# @with_state_model(GraphState)
# def web_search_node(state: GraphState):
#     search_query = f"{state.rewritten_question} \"{REPO_IDENTIFIER}\""
#     new_docs = web_search_tool.invoke({"query": search_query})

#     # Keep only results whose URL is actually the target repo
#     repo_path = REPO_IDENTIFIER.lower()
#     relevant_docs = [
#         d for d in new_docs
#         if f"github.com/{repo_path}" in d.get("url", "").lower()
#     ]

#     web_results = [d["content"] for d in relevant_docs]
#     updated_documents = state.documents + web_results
#     return {"documents": updated_documents, "retrieval_num": state.retrieval_num + 1}
########

@with_state_model(GraphState)
def web_search_node(state: GraphState):
    search_query = state.rewritten_question
    if state.repo_scoped_search:
        search_query = f'{search_query} "{REPO_IDENTIFIER}"'

    new_docs = web_search_tool.invoke({"query": search_query})

    if state.repo_scoped_search:
        repo_path = REPO_IDENTIFIER.lower()
        relevant_docs = [d for d in new_docs if f"github.com/{repo_path}" in d.get("url", "").lower()]
    else:
        relevant_docs = new_docs

    web_results = [d["content"] for d in relevant_docs]
    updated_documents = state.documents + web_results
    return {"documents": updated_documents, "retrieval_num": state.retrieval_num + 1}

@with_state_model(GraphState)
def search_mode_node(state:GraphState):
    return state.search_mode

@with_state_model(GraphState)
def relevant_documents_validation_node(state: GraphState):
    if state.documents:
        ## we have relevant documents
        return "knowledge_extraction" 
    elif (state.search_mode == "vectorstore" and state.retrieval_num >= MAX_RETRIEVALS):  
         ### we don't have relevant documents
        # and we reached the maximum number of retrievals
        return "max_db_search"
    elif (state.search_mode == "websearch" and state.retrieval_num >= MAX_RETRIEVALS):
         # we don't have relevant documents
        # and we reached the maximum number of websearches
        return "max_websearch"
    else:
        # we don't have relevant documents
        # so we retry the search
        return state.search_mode