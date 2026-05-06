"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and calculate total cost from the state trace."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # Calculate total cost from all trace events
    total_cost = sum(
        event.get("payload", {}).get("cost_usd", 0.0) 
        for event in state.trace
    )
    
    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        notes=f"Total iterations: {state.iteration}"
    )
    return state, metrics
