from langgraph.graph import START, END, StateGraph
from node import ( 
    retriever_node, generation_node, db_query_rewriting_node, GraphState, with_state_model, 
    answer_evaluation_node, query_feedback_node, generation_feedback_node, give_up_node, router_node,search_mode_node,
    filter_relevant_documents_node, knowledge_extractor_node, simple_question_node, websearch_query_rewriting_node, web_search_node,
    relevant_documents_validation_node
)

pipeline = StateGraph(GraphState)

pipeline.add_node('retriever_node', retriever_node)
pipeline.add_node('generator_node', generation_node)
pipeline.add_node('db_query_rewrite_node', db_query_rewriting_node)
pipeline.add_node('query_feedback_node', query_feedback_node)
pipeline.add_node('generation_feedback_node', generation_feedback_node)
pipeline.add_node('give_up_node', give_up_node)
pipeline.add_node('filtered_docs_node', filter_relevant_documents_node)
pipeline.add_node('extract_knowledge_node', knowledge_extractor_node)
pipeline.add_node('simple_question_node', simple_question_node)
pipeline.add_node('websearch_query_rewriting_node', websearch_query_rewriting_node)
pipeline.add_node('web_search_node', web_search_node)

pipeline.add_conditional_edges(
    START,
    router_node,
    {
        "vectorstore": 'db_query_rewrite_node',
        "websearch": 'websearch_query_rewriting_node',
        "QA_LM": 'simple_question_node'
    }
)

pipeline.add_conditional_edges(
    'generator_node',
    answer_evaluation_node,
    {
        "useful": END,
        "not relevant": 'query_feedback_node',
        "hallucination": 'generation_feedback_node',
        "max_generation_reached": 'give_up_node'
    }
)

pipeline.add_conditional_edges(
    'query_feedback_node', 
    search_mode_node,
    {
        "vectorstore": 'db_query_rewrite_node',
        "websearch": 'websearch_query_rewriting_node',
    }
)

pipeline.add_conditional_edges(
    'filtered_docs_node',
    relevant_documents_validation_node, 
    {
        "knowledge_extraction": 'extract_knowledge_node',
        "websearch": 'websearch_query_rewriting_node',
        "vectorstore": 'db_query_rewrite_node',
        "max_db_search": 'websearch_query_rewriting_node',
        "max_websearch": 'give_up_node'
    }
)

# pipeline.add_edge(START, 'db_query_rewrite_node')
pipeline.add_edge('db_query_rewrite_node', 'retriever_node')
pipeline.add_edge('retriever_node', 'filtered_docs_node')
# pipeline.add_edge('filtered_docs_node', 'extract_knowledge_node')
pipeline.add_edge('extract_knowledge_node', 'generator_node')
# pipeline.add_edge('query_feedback_node', 'db_query_rewrite_node')
pipeline.add_edge('websearch_query_rewriting_node', 'web_search_node')
pipeline.add_edge('web_search_node', 'filtered_docs_node')
pipeline.add_edge('generation_feedback_node', 'generator_node')
pipeline.add_edge('simple_question_node', END)
pipeline.add_edge('give_up_node', END)

rag_pipeline = pipeline.compile()

print(pipeline.compile().get_graph().draw_mermaid())

if __name__ == "__main__":
    # inputs = GraphState(question="What does the estimate_loss function do?")

    inputs = GraphState(question="How does this repo implement LoRA fine-tuning?")
    # inputs = GraphState(question="How is the stock data fetched and processed in this repo?")

    for output in rag_pipeline.stream(inputs, stream_mode='updates'):
        for key, value in output.items():
            print(f"Node: {key}")
            # print(f"Output: {value}")
    print(value["generation"])

######## Previous Queries ######
#####1.  "What does the GPT class do?"
#####2. What does the estimate_loss function do
#######################################################################################
#     Good test queries to try, roughly by difficulty:

# Simple / conceptual:

# "What does the GPT class do?"
# "How is the model configured?" (tests if it finds GPTConfig)

# Specific implementation details (better tests of retrieval precision):

# "How does the causal self-attention mask work in this repo?"
# "What optimizer and learning rate schedule does train.py use?"
# "How does this repo implement weight tying between the embedding and output layers?"
# "What does the estimate_loss function do?"
# "How is gradient accumulation implemented in the training loop?"

# Cross-file/harder (good stress test for retrieval + multi-chunk reasoning):

# "How does the training script use the GPT model class?"
# "How does sample.py load and use a saved checkpoint?"

## "How does this repo implement LoRA fine-tuning?" — nanoGPT doesn't have LoRA — or
#  "How does this repo handle multi-modal image+text inputs?").