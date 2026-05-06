"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Mock search client for lab purposes."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return simulated search results."""
        
        return [
            SourceDocument(
                title=f"Mock Source {i+1} for: {query}",
                url=f"https://example.com/mock-{i+1}",
                snippet=f"This is a simulated search result for the research query about {query}. It contains key information points {i+1}."
            )
            for i in range(max_results)
        ]
