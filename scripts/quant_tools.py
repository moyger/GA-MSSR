"""Custom MCP tools for the GA-MSSR Quant Strategy Development Team.

Wraps the existing monitoring infrastructure (monitoring/readers.py,
monitoring/config.py) as structured MCP tools for agent consumption.

Usage:
    from quant_tools import quant_mcp_server
    options = ClaudeAgentOptions(mcp_servers={"quant": quant_mcp_server}, ...)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Lazily import monitoring to avoid import errors if monitoring isn't on path
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.config import MonitorConfig, BotDescriptor
from monitoring import readers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MONITOR_CONFIG = MonitorConfig(project_root=PROJECT_ROOT)
_BOT_MAP: dict[str, BotDescriptor] = {b.bot_id: b for b in _MONITOR_CONFIG.bots}


def _get_bots(bot_id: str) -> list[BotDescriptor]:
    if bot_id == "all":
        return list(_MONITOR_CONFIG.bots)
    bot = _BOT_MAP.get(bot_id)
    if not bot:
        raise ValueError(f"Unknown bot_id '{bot_id}'. Valid: {list(_BOT_MAP.keys())}")
    return [bot]


def _json_response(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]}


# ---------------------------------------------------------------------------
# Tool 1: get_bot_status
# ---------------------------------------------------------------------------

@tool(
    "get_bot_status",
    "Get live status of GA-MSSR trading bots: process state, position, "
    "heartbeat, model age, daily PnL, and recent trades.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "description": "Bot to check: 'futures', 'bybit', 'forex', or 'all'",
                "enum": ["futures", "bybit", "forex", "all"],
            }
        },
        "required": ["bot_id"],
    },
)
async def get_bot_status(args: dict[str, Any]) -> dict[str, Any]:
    bots = _get_bots(args["bot_id"])
    results = []
    for bot in bots:
        status = readers.get_bot_status(bot, PROJECT_ROOT)
        entry = {
            "bot_name": bot.name,
            "bot_id": bot.bot_id,
            "alive": status["alive"],
            "pid": status["pid"],
            "position": status["position"],
            "position_label": status["position_label"],
            "heartbeat_age_sec": status["heartbeat_age_sec"],
            "model_age_hours": status["model_age_hours"],
            "log_freshness_sec": status["log_freshness_sec"],
            "last_bar": status["last_bar"],
            "trades_today_count": len(status["trades_today"]),
            "trades_today": status["trades_today"][-5:],  # last 5
        }
        # Add daily PnL from heartbeat if available
        hb = status.get("heartbeat")
        if hb:
            entry["daily_pnl"] = hb.get("daily_pnl")
            entry["bars_today"] = hb.get("bars_today")
            entry["uptime_sec"] = hb.get("uptime_sec")
        results.append(entry)
    return _json_response(results)


# ---------------------------------------------------------------------------
# Tool 2: get_trade_log
# ---------------------------------------------------------------------------

@tool(
    "get_trade_log",
    "Read trade log entries for a bot. Returns JSONL trade records.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "enum": ["futures", "bybit", "forex"],
            },
            "last_n": {
                "type": "integer",
                "description": "Return last N trades (default 20)",
            },
        },
        "required": ["bot_id"],
    },
)
async def get_trade_log(args: dict[str, Any]) -> dict[str, Any]:
    bot = _BOT_MAP.get(args["bot_id"])
    if not bot or not bot.trade_log_file:
        return _json_response({"error": f"No trade log for '{args['bot_id']}'"})

    last_n = args.get("last_n", 20)
    trade_path = PROJECT_ROOT / bot.trade_log_file

    if not trade_path.exists():
        return _json_response({"trades": [], "message": "Trade log file not found"})

    trades: list[dict] = []
    try:
        with open(trade_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
    except Exception as e:
        return _json_response({"error": str(e)})

    return _json_response({
        "bot_id": args["bot_id"],
        "total_trades": len(trades),
        "trades": trades[-last_n:],
    })


# ---------------------------------------------------------------------------
# Tool 3: get_model_state
# ---------------------------------------------------------------------------

@tool(
    "get_model_state",
    "Read the trained model state: GA weights, rule parameters, SSR fitness, "
    "training timestamp, max drawdown.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "enum": ["futures", "bybit", "forex"],
            }
        },
        "required": ["bot_id"],
    },
)
async def get_model_state(args: dict[str, Any]) -> dict[str, Any]:
    bot = _BOT_MAP.get(args["bot_id"])
    if not bot:
        return _json_response({"error": f"Unknown bot_id '{args['bot_id']}'"})

    model_path = PROJECT_ROOT / bot.model_state_file
    model = readers.read_json_file(model_path)
    if not model:
        return _json_response({"error": "Model state file not found or unreadable"})

    age_hours = readers.get_model_age_hours(model)
    model["model_age_hours"] = age_hours
    return _json_response(model)


# ---------------------------------------------------------------------------
# Tool 4: get_risk_state
# ---------------------------------------------------------------------------

@tool(
    "get_risk_state",
    "Read risk management state for the futures bot: EOD high balance, "
    "trailing floor, daily PnL.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "enum": ["futures"],
                "description": "Currently only 'futures' has a risk state file",
            }
        },
        "required": ["bot_id"],
    },
)
async def get_risk_state(args: dict[str, Any]) -> dict[str, Any]:
    risk_path = PROJECT_ROOT / "live_futures" / "state" / "fundednext" / "risk_state.json"
    data = readers.read_json_file(risk_path)
    if not data:
        return _json_response({"error": "Risk state file not found"})
    return _json_response(data)


# ---------------------------------------------------------------------------
# Tool 5: read_heartbeat
# ---------------------------------------------------------------------------

@tool(
    "read_heartbeat",
    "Read heartbeat status: uptime, bars processed today, last bar time, "
    "current position, daily PnL.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "enum": ["futures", "bybit", "forex"],
            }
        },
        "required": ["bot_id"],
    },
)
async def read_heartbeat(args: dict[str, Any]) -> dict[str, Any]:
    bot = _BOT_MAP.get(args["bot_id"])
    if not bot:
        return _json_response({"error": f"Unknown bot_id '{args['bot_id']}'"})

    hb_path = PROJECT_ROOT / bot.heartbeat_file
    data = readers.read_json_file(hb_path)
    if not data:
        return _json_response({"error": "Heartbeat file not found"})

    age_sec = readers.get_heartbeat_age_sec(data)
    data["heartbeat_age_sec"] = age_sec
    return _json_response(data)


# ---------------------------------------------------------------------------
# Tool 6: check_process_health
# ---------------------------------------------------------------------------

@tool(
    "check_process_health",
    "Check if bot processes are running. Returns PID, alive status, and "
    "launchd daemon status.",
    {
        "type": "object",
        "properties": {
            "bot_id": {
                "type": "string",
                "enum": ["futures", "bybit", "forex", "all"],
            }
        },
        "required": ["bot_id"],
    },
)
async def check_process_health(args: dict[str, Any]) -> dict[str, Any]:
    bots = _get_bots(args["bot_id"])
    results = []
    for bot in bots:
        pid = readers.is_process_running(bot.process_keyword)

        # Check PID lockfile
        pid_files = {
            "futures": "/tmp/ga_mssr_futures.pid",
            "bybit": "/tmp/ga_mssr_live.pid",
            "forex": "/tmp/ga_mssr_forex.pid",
        }
        pid_file = pid_files.get(bot.bot_id, "")
        lockfile_pid = None
        if pid_file and os.path.exists(pid_file):
            try:
                lockfile_pid = int(Path(pid_file).read_text().strip())
            except (ValueError, OSError):
                pass

        entry = {
            "bot_id": bot.bot_id,
            "bot_name": bot.name,
            "process_keyword": bot.process_keyword,
            "pid": pid,
            "alive": pid is not None,
            "lockfile_pid": lockfile_pid,
        }

        # Check launchd for Bybit bot
        if bot.bot_id == "bybit":
            try:
                result = subprocess.run(
                    ["launchctl", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.split("\n"):
                    if "ga-mssr" in line:
                        entry["launchd_status"] = line.strip()
                        break
            except Exception:
                entry["launchd_status"] = "unknown"

        results.append(entry)

    return _json_response(results)


# ---------------------------------------------------------------------------
# Assemble MCP server
# ---------------------------------------------------------------------------

quant_mcp_server = create_sdk_mcp_server(
    name="quant",
    version="1.0.0",
    tools=[
        get_bot_status,
        get_trade_log,
        get_model_state,
        get_risk_state,
        read_heartbeat,
        check_process_health,
    ],
)
