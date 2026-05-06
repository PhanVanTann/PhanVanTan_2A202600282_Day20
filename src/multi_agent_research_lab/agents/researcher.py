from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.search_client = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        
        # 1. Search for information
        query = state.request.query
        sources = self.search_client.search(query, max_results=state.request.max_sources)
        state.sources.extend(sources)
        
        # 2. Synthesize notes using LLM
        source_text = "\n".join([f"- {s.title}: {s.snippet}" for s in sources])
        system_prompt = "You are a researcher. Synthesize the following search results into concise research notes."
        user_prompt = f"Query: {query}\n\nSearch Results:\n{source_text}"
        
        response = self.llm.complete(system_prompt, user_prompt)
        state.research_notes = response.content
        
        state.add_trace_event("research_completed", {
            "num_sources": len(sources),
            "cost_usd": response.cost_usd
        })
        
        return state
