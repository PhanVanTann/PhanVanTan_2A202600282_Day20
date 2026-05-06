"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Mock LLM client for lab purposes when no API key is available."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # No real client initialization needed for mock

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a simulated model completion that progresses through the workflow."""
        
        user_lower = user_prompt.lower()
        
        # Smart Mock logic: Only look at the history in user_prompt to decide next step
        if "supervisor" in system_prompt.lower():
            if "writer" in user_lower:
                content = "DONE"
            elif "critic" in user_lower:
                content = "WRITER"
            elif "analyst" in user_lower:
                content = "CRITIC"
            elif "researcher" in user_lower:
                content = "ANALYST"
            else:
                content = "RESEARCHER"
        elif "researcher" in system_prompt.lower():
            content = "Here are some mock research notes about the query: 1. GraphRAG uses knowledge graphs. 2. It improves retrieval context."
        elif "analyst" in system_prompt.lower():
            content = "Analysis: GraphRAG is superior to flat RAG for complex, multi-hop queries because it preserves global context."
        elif "critic" in system_prompt.lower():
            content = "Critic Review: The research is solid, but we could add more details about performance benchmarks vs ChromaDB."
        elif "writer" in system_prompt.lower():
            content = "# GraphRAG Research Report\n\nGraphRAG represents a significant leap in RAG technology by integrating knowledge graphs for better context retrieval."
        elif "research assistant" in system_prompt.lower():
            content = f"Bản tóm tắt nghiên cứu (Single-Agent) cho câu hỏi: '{user_prompt[:30]}...'. Kết quả cho thấy đây là một lĩnh vực đang phát triển mạnh mẽ với nhiều tiềm năng ứng dụng trong thực tế."
        else:
            content = f"Hệ thống đã nhận được yêu cầu: {user_prompt[:50]}..."

        input_tokens = (len(system_prompt) + len(user_prompt)) // 4
        output_tokens = len(content) // 4
        
        # Simulate cost: $0.15/1M input, $0.60/1M output
        simulated_cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=simulated_cost,
        )
