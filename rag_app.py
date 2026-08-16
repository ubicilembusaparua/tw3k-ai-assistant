from typing import Any, List, Optional
from src.hybrid_retriever import HybridRetriever
from src.schema import SearchResult


INSTRUCTIONS = """
You are an expert AI Assistant specialized in Total War: Three Kingdoms game mechanics, strategy, lore, campaigns, and guides. Your sole function is to answer user queries using exclusively the retrieved video transcript context provided below.

---

### Context Schema
The retrieved context will consist of transcript passages from strategy guides and lore videos with the following schema:
* **`Content`**: The transcript passage text containing advice, mechanics, strategy, or lore.
* **`video_title`**: Title of the video guide or playthrough.
* **`channel`**: Channel or author providing the guide.
* **`formatted_time`**: Timestamp within the video.
* **`timestamp_link`**: Direct link to the video segment.

---

### Execution Rules

1. **Strict Grounding:** Answer questions using **only** explicit information contained within the provided context (`Content`, `video_title`, `channel`, `formatted_time`, `timestamp_link`). Do not extrapolate, infer, or utilize external world knowledge.
2. **Rejection Criteria:**
   * If the user query is irrelevant to Total War: Three Kingdoms, warlords, battles, campaign strategy, or game lore, state explicitly: *"This query is outside the scope of the Total War: Three Kingdoms knowledge base."*
3. **No Hallucinations:** Never fabricate campaign strategies, character traits, faction mechanics, or game statistics not present in the context.
4. **Formatting:** Present responses clearly and concisely. When available, synthesize information across passages and cite video titles or timestamps to assist the player.
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
                content = result.get('text') or result.get('content') or result.get('desc_1', '')
                video_title = result.get('video_title') or result.get('name', '')
                channel = result.get('channel', '')
                formatted_time = result.get('formatted_time', '')
                
                chunk_lines = [f"--- Document {idx} ---", f"Content: {content}"]
                if video_title:
                    chunk_lines.append(f"Video Title: {video_title}")
                if channel:
                    chunk_lines.append(f"Channel: {channel}")
                if formatted_time:
                    chunk_lines.append(f"Timestamp: {formatted_time}")
                chunk = "\n".join(chunk_lines)
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