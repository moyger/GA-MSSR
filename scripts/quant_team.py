#!/usr/bin/env python3
"""
GA-MSSR Quant Strategy Development Team

Multi-agent system using the Claude Agent SDK for automated quant research,
strategy development, risk monitoring, and bot operations.

Usage:
    # Single query
    .venv/bin/python scripts/quant_team.py "check all bot health"
    .venv/bin/python scripts/quant_team.py "run walk-forward on NQ 15min"
    .venv/bin/python scripts/quant_team.py "investigate why NQ bot lost money today"
    .venv/bin/python scripts/quant_team.py "develop a VWAP indicator"

    # Interactive REPL
    .venv/bin/python scripts/quant_team.py

Requires: pip install claude-agent-sdk
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_tools import quant_mcp_server
from quant_agents import ALL_AGENTS


# ── Orchestrator system prompt ───────────────────────────────────────────

ORCHESTRATOR_PROMPT = """\
You are the lead quant of the GA-MSSR trading team. You coordinate 6 \
specialized agents to accomplish quantitative trading tasks.

YOUR TEAM:
1. research_analyst — Explores data, runs backtests, analyzes walk-forward \
results, identifies patterns. Model: Sonnet.
2. strategy_developer — Modifies indicators, rules, signal engine code. \
Runs tests after changes. Model: Opus.
3. risk_manager — Monitors live bot health, checks prop firm compliance, \
analyzes PnL and trade logs. Model: Sonnet.
4. quant_researcher — Runs walk-forward validations, parameter sweeps, \
filter experiments (CPU-intensive, 5-30 min). Model: Sonnet.
5. devops_operator — Manages bot processes, launchd daemons, restarts, \
log inspection. Model: Haiku (fast).
6. code_reviewer — Reviews code changes, runs pytest, checks regressions. \
Model: Sonnet.

DELEGATION RULES:
- Use the Task tool to delegate work to the appropriate agent by name.
- For multi-step workflows, delegate sequentially: e.g., strategy_developer \
writes code → code_reviewer validates → quant_researcher runs walk-forward.
- Run agents in parallel when tasks are independent (e.g., risk_manager + \
devops_operator for health checks).
- Always summarize subagent results into a coherent response.
- If an agent reports an error, determine if another agent should investigate.

WORKFLOW EXAMPLES:
1. "Develop a new indicator" → strategy_developer (implement + test) → \
code_reviewer (review) → quant_researcher (walk-forward validation)
2. "Why did NQ bot lose money today?" → risk_manager (PnL, trades) + \
research_analyst (signal analysis) in parallel
3. "Check system health" → devops_operator (processes) + risk_manager \
(risk metrics) in parallel
4. "Compare 5min vs 15min" → quant_researcher (run both walk-forwards) → \
research_analyst (interpret results)
5. "Restart the futures bot" → devops_operator (check position, stop, start)

PROJECT ROOT: /Users/karloestrada/algos/ga-mssr/
This system trades real money across 3 instruments. Safety and accuracy \
are paramount.

You also have direct access to custom MCP tools for quick status checks \
without delegating. Use them for simple queries; delegate to agents for \
complex analysis."""


# ── Build options ────────────────────────────────────────────────────────

def build_options(max_turns: int = 50) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions with all agents and MCP tools."""
    return ClaudeAgentOptions(
        system_prompt=ORCHESTRATOR_PROMPT,
        permission_mode="acceptEdits",
        cwd=str(PROJECT_ROOT),
        model="claude-sonnet-4-6",
        mcp_servers={"quant": quant_mcp_server},
        allowed_tools=[
            # Built-in tools for the orchestrator
            "Read", "Glob", "Grep", "Bash", "Task",
            # Custom MCP tools (orchestrator can use directly)
            "mcp__quant__get_bot_status",
            "mcp__quant__get_trade_log",
            "mcp__quant__get_model_state",
            "mcp__quant__get_risk_state",
            "mcp__quant__read_heartbeat",
            "mcp__quant__check_process_health",
        ],
        agents=ALL_AGENTS,
        setting_sources=["project"],  # Load CLAUDE.md
        max_turns=max_turns,
    )


# ── Audit logging ────────────────────────────────────────────────────────

AUDIT_LOG = PROJECT_ROOT / "logs" / "agent_audit.jsonl"


def _audit_log(event: str, detail: str = "") -> None:
    """Append an audit entry to the agent audit log."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail,
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Message printer ──────────────────────────────────────────────────────

def _print_message(message) -> None:
    """Extract and print text content from a message."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text, flush=True)
    elif isinstance(message, ResultMessage):
        if message.result:
            print(message.result, flush=True)
        cost = message.total_cost_usd
        turns = message.num_turns
        duration = message.duration_ms / 1000 if message.duration_ms else 0
        print(f"\n[Turns: {turns} | Time: {duration:.1f}s | "
              f"Cost: ${cost:.4f}]" if cost else "", flush=True)


# ── Single query mode ────────────────────────────────────────────────────

async def run_single_query(prompt: str) -> None:
    """Run a single query through the orchestrator."""
    _audit_log("query_start", prompt[:200])
    options = build_options()

    async for message in query(prompt=prompt, options=options):
        _print_message(message)

    _audit_log("query_end", prompt[:200])


# ── Interactive REPL mode ────────────────────────────────────────────────

async def run_interactive() -> None:
    """Run an interactive multi-turn session."""
    options = build_options()

    print("=" * 60)
    print("  GA-MSSR Quant Strategy Development Team")
    print("=" * 60)
    print()
    print("  Agents: research_analyst, strategy_developer, risk_manager,")
    print("          quant_researcher, devops_operator, code_reviewer")
    print()
    print("  Type 'quit' to exit. Type 'help' for example queries.")
    print("=" * 60)
    print()

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("quant> ").strip()

                if user_input.lower() in ("quit", "exit", "q"):
                    print("Shutting down quant team.")
                    break

                if user_input.lower() == "help":
                    _print_help()
                    continue

                if not user_input:
                    continue

                _audit_log("interactive_query", user_input[:200])
                await client.query(user_input)

                async for message in client.receive_response():
                    _print_message(message)

                print()

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                print("\nExiting.")
                break


def _print_help() -> None:
    print("""
Example queries:
  "Check all bot health and risk status"
  "What's the current position on the futures bot?"
  "Show me today's trades for the Bybit bot"
  "Run walk-forward on NQ 15min data"
  "Compare ER p=14 t=0.40 vs p=20 t=0.35 on NQ"
  "Why did the NQ bot lose money today?"
  "Develop a VWAP indicator and validate it"
  "Review the changes in strategies/indicators.py"
  "Restart the forex bot"
  "Is the model stale on any bot?"
""")


# ── Entry point ──────────────────────────────────────────────────────────

async def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        await run_single_query(prompt)
    else:
        await run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
