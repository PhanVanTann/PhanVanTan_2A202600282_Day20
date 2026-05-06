"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Reviews research and analysis for quality and missing info."""

    name = "critic"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate trace with review notes and potentially flag errors."""
        
        system_prompt = (
            "You are a critical reviewer. Evaluate the current research and "
            "analysis for accuracy, depth, and missing information."
        )
        user_prompt = f"Research: {state.research_notes}\nAnalysis: {state.analysis_notes}"
        
        response = self.llm.complete(system_prompt, user_prompt)
        state.add_trace_event("critic_review", {"review": response.content})
        
        return state
