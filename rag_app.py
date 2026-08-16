from typing import Any, List, Optional
from src.hybrid_retriever import HybridRetriever
from src.schema import SearchResult


INSTRUCTIONS = """
You are a precise coffee review analysis assistant. Your sole function is to answer user queries using exclusively the retrieved coffee review context provided below.

---

### Context Schema
The retrieved context will consist of data entries with the following schema:
* **`name`**: Name of the coffee blend/single origin.
* **`roast`**: Roast profile (`Light`, `Medium-Light`, `Medium`, `Medium-Dark`, `Dark`).
* **`loc_country`**: Country where the roaster is located.
* **`origin_1`**: Origin location of the coffee beans.
* **`origin_2`**: Second origin location of the coffee beans.
* **`rating`**: Score or rating assigned to the coffee.
* **`desc_1`**: First review text excerpt.
* **`desc_2`**: Second review text excerpt.
* **`desc_3`**: Third review text excerpt.

---

### Execution Rules

1. **Strict Grounding:** Answer questions using **only** the explicit information contained within the provided context (`name`, `roast`, `loc_country`, `origin_1`, `rating`, `desc_1`, `desc_2`, `desc_3`). Do not extrapolate, infer, or utilize external world knowledge.
2. **Rejection Criteria:**
   * If the user query is irrelevant to coffee, coffee roasters, origins, ratings, or reviews, state explicitly: *"This query is outside the scope of the coffee review database."*
3. **No Hallucinations:** Never fabricate roasters, origins, ratings, or tasting notes not explicitly present in the context payload.
4. **Formatting:** Present responses concisely. Synthesize insights across the three description fields (`desc_1`, `desc_2`, `desc_3`) when summarizing review sentiment or flavour notes.
""".strip()

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

class RAGBase():
    
    def __init__(
            self,
            index: Any,
            llm_client: Any = None,
            instructions: str = INSTRUCTIONS,
            prompt_template: str = PROMPT_TEMPLATE,
            model: str = "gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str, num_results: int = 5) -> List[Any]:
        """Performs hybrid retrieval combining BM25 keyword search and dense vector search via Reciprocal Rank Fusion (RRF)."""
        if hasattr(self.index, "search"):
            try:
                return self.index.search(query, top_k=num_results)
            except TypeError:
                return self.index.search(query, num_results=num_results)
        elif hasattr(self.index, "hybrid_search"):
            return self.index.hybrid_search(query, top_k=num_results)
        return []

    def build_context(self, search_results):
        context_chunks = []

        for idx, result in enumerate(search_results, start=1):
            if hasattr(result, "chunk"):
                chunk_obj = result.chunk
                content = chunk_obj.content
                meta = chunk_obj.metadata or {}
                chunk_lines = [f"--- Document {idx} ---", f"Content: {content}"]
                for k, v in meta.items():
                    if v:
                        chunk_lines.append(f"{k}: {v}")
                chunk = "\n".join(chunk_lines)
            elif isinstance(result, dict):
                chunk = (
                    f"--- Document {idx} ---\n"
                    f"Coffee Name: {result.get('name', '')}\n"
                    f"Origin 1: {result.get('origin_1', '')}\n"
                    f"Origin 2: {result.get('origin_2', '')}\n"
                    f"Description 1: {result.get('desc_1', '')}\n"
                    f"Description 2: {result.get('desc_2', '')}\n"
                    f"Description 3: {result.get('desc_3', '')}\n"
                    f"Roast Level: {result.get('roast', '')}\n"
                    f"Quality Rating: {result.get('rating', '')}\n"
                    f"Location/Country: {result.get('loc_country', '')}\n"
                )
            else:
                chunk = f"--- Document {idx} ---\n{str(result)}"
            context_chunks.append(chunk)

        return "\n\n".join(context_chunks)
    
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query,
            context=context
        )
    
    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
       
        response = self.llm_client.responses.create(
            model = self.model,
            input = input_messages,
        )

        return response
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response