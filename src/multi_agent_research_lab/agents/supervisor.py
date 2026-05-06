from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        
        # 1. Guardrail: Enforce max iterations
        if state.iteration >= self.settings.max_iterations:
            state.record_route("DONE")
            state.add_trace_event("max_iterations_reached", {"iteration": state.iteration})
            return state

        # 2. Routing logic
        system_prompt = (
            "You are a research supervisor. Based on the current state, "
            "decide the next step: RESEARCHER, ANALYST, CRITIC, WRITER, or DONE."
        )
        user_prompt = f"Query: {state.request.query}\nHistory: {state.route_history}"
        
        response = self.llm.complete(system_prompt, user_prompt)
        next_route = response.content.strip().upper()
        
        # 3. Fallback: Validate output
        valid_routes = ["RESEARCHER", "ANALYST", "CRITIC", "WRITER", "DONE"]
        if next_route not in valid_routes:
            next_route = "RESEARCHER"
            
        state.record_route(next_route)
        state.add_trace_event("supervisor_decision", {
            "next_route": next_route,
            "cost_usd": response.cost_usd
        })
        
        return state
