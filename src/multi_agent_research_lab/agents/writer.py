from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        
        if not state.analysis_notes:
            state.errors.append("Writer ran but no analysis notes found.")
            return state
            
        system_prompt = (
            "You are a technical writer. Write a clear, structured final report "
            "based on the following analysis notes. Use Markdown."
        )
        user_prompt = f"Analysis Notes:\n{state.analysis_notes}"
        
        response = self.llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content
        
        state.add_trace_event("writing_completed", {
            "answer_length": len(response.content),
            "cost_usd": response.cost_usd
        })
        
        return state
