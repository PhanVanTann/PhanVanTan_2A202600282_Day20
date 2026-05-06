"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    llm = LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    # Simple single-agent implementation
    response = llm.complete(
        "You are a helpful research assistant.",
        f"Research this query and give a concise summary: {query}"
    )
    state.final_answer = response.content
    state.add_trace_event("baseline_complete", {"cost_usd": response.cost_usd})
    
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
        
        # Save trace to file
        store = LocalArtifactStore()
        path = store.write_json("last_run_trace.json", result.model_dump())
        
        console.print(Panel.fit(f"Trace saved to: {path}", title="Execution Success", style="green"))
        console.print(result.model_dump_json(indent=2))
        
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run benchmark: Single-Agent vs Multi-Agent."""

    _init()
    console.print("[bold blue]Starting Benchmark...[/bold blue]")

    # 1. Run Baseline
    def run_baseline_internal(q: str) -> ResearchState:
        llm = LLMClient()
        state = ResearchState(request=ResearchQuery(query=q))
        resp = llm.complete("You are a researcher.", q)
        state.final_answer = resp.content
        state.add_trace_event("baseline", {"cost_usd": resp.cost_usd})
        return state

    # 2. Run Multi-Agent
    def run_multi_internal(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    _, m_baseline = run_benchmark("Baseline (Single)", query, run_baseline_internal)
    console.print("Baseline completed.")
    
    _, m_multi = run_benchmark("Multi-Agent", query, run_multi_internal)
    console.print("Multi-Agent completed.")

    # 3. Render and Save Report
    report = render_markdown_report([m_baseline, m_multi])
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report)
    
    console.print(Panel.fit(f"Report saved to: {report_path}", title="Benchmark Results"))
    console.print(report)


if __name__ == "__main__":
    app()
