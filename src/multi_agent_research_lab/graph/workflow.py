from langgraph.graph import StateGraph, END

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph using LangGraph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.critic = CriticAgent()
        self.writer = WriterAgent()

    def build(self):
        """Create a LangGraph graph."""
        
        workflow = StateGraph(ResearchState)

        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("critic", self.critic.run)
        workflow.add_node("writer", self.writer.run)

        workflow.set_entry_point("supervisor")

        def route_decision(state: ResearchState) -> str:
            next_route = state.route_history[-1].lower()
            if next_route == "done":
                return END
            return next_route

        workflow.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "critic": "critic",
                "writer": "writer",
                END: END
            }
        )

        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("critic", "supervisor")
        workflow.add_edge("writer", "supervisor")

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state with guardrails."""
        
        app = self.build()
        
        # Guardrail: Set recursion limit based on max iterations
        # (Supervisor -> Worker -> Supervisor counts as multiple steps)
        config = {"recursion_limit": self.supervisor.settings.max_iterations * 3}
        
        try:
            result = app.invoke(state, config=config)
            
            if isinstance(result, dict):
                return ResearchState(**result)
            return result
            
        except Exception as e:
            state.errors.append(f"Workflow execution failed: {str(e)}")
            state.add_trace_event("workflow_error", {"error": str(e)})
            return state
