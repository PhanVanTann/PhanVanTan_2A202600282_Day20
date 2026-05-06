import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_runs_successfully() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    
    assert len(result.route_history) == 1
    assert result.route_history[0] in ["RESEARCHER", "ANALYST", "WRITER", "DONE"]
    assert result.iteration == 1
