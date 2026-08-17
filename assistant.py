import sys

from dotenv import load_dotenv
from rag_app import RAGBase
from openai import OpenAI
from query_rewriter import QueryRewriter

from metrics import RAGWithMetrics

def create_assistant(*, rerank: bool = False):
    load_dotenv()

    return RAGWithMetrics.from_src(rerank=rerank, llm_client=OpenAI(), query_rewriter=QueryRewriter(OpenAI()))

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How to manage food?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)
