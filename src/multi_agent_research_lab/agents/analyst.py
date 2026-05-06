from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        
        if not state.research_notes:
            state.errors.append("Analyst ran but no research notes found.")
            return state
            
        system_prompt = "You are a senior analyst. Extract key insights and patterns from the following research notes."
        user_prompt = f"Research Notes:\n{state.research_notes}"
        
        response = self.llm.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content
        
        state.add_trace_event("analysis_completed", {
            "notes_length": len(response.content),
            "cost_usd": response.cost_usd
        })
        
        return state
