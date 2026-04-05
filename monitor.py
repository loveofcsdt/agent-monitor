#!/usr/bin/env python3
"""
Agent Monitor — Dashboard for OpenClaw and Claude Code instances.

Usage:
    python toolkits/agent_monitor/monitor.py [--port 9900] [--config config.yaml]

Monitors local and remote machines for running OpenClaw/Claude Code processes,
shows task summaries (via Gemini Flash), context usage, and call hierarchy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_PORT = 9900
SUMMARY_CACHE_TTL = 180  # seconds — avoid hammering Gemini
GEMINI_MODEL = "gemini-2.5-flash"

# Known Claude model context windows (tokens)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5-20250514": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-opus": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 200_000

CLAUDE_HOME = Path.home() / ".claude"


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class AgentNode:
    id: str
    type: str  # "openclaw" | "claude_code"
    pid: int
    ppid: int
    status: str = "running"
    working_dir: str = ""
    command: str = ""
    summary: str = ""
    context_percent: Optional[float] = None
    context_tokens: Optional[int] = None
    context_window: Optional[int] = None
    model: str = ""
    session_id: str = ""
    children: list[AgentNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "pid": self.pid,
            "ppid": self.ppid,
            "status": self.status,
            "working_dir": self.working_dir,
            "command": self.command,
            "summary": self.summary,
            "context_percent": self.context_percent,
            "context_tokens": self.context_tokens,
            "context_window": self.context_window,
            "model": self.model,
            "session_id": self.session_id,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class MachineStatus:
    name: str
    host: str  # "local" or SSH target
    status: str = "connected"
    error: str = ""
    agents: list[AgentNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "status": self.status,
            "error": self.error,
            "agents": [a.to_dict() for a in self.agents],
        }


# ──────────────────────────────────────────────
# Process Detection Helpers
# ──────────────────────────────────────────────

def _is_noise(cmd: str) -> bool:
    """Filter out non-agent processes that happen to match keywords."""
    if "grep" in cmd:
        return True
    if "agent_monitor" in cmd or "monitor.py" in cmd:
        return True
    # Chrome / Electron helper processes (GPU, renderer, utility, storage, audio, etc.)
    if re.search(r"--type=(gpu-process|renderer|utility|zygote)", cmd):
        return True
    # Chrome main process with --user-data-dir (OpenClaw's headless browser)
    if "Google Chrome" in cmd and "--user-data-dir" in cmd:
        return True
    # Electron helper subprocesses (Genspark Claw Helper GPU/Renderer/etc.)
    if "Genspark Claw Helper" in cmd:
        return True
    return False


def is_claude_code(cmd: str) -> bool:
    if _is_noise(cmd):
        return False
    # Match the claude CLI binary, claude-agent-acp, or claude-bridge
    patterns = [
        r"(?:^|/)claude\s",
        r"(?:^|/)claude$",
        r"claude-agent-acp",
        r"claude-bridge",
        r"/\.claude/local/",
    ]
    return any(re.search(p, cmd) for p in patterns)


def is_openclaw(cmd: str) -> bool:
    if _is_noise(cmd):
        return False
    patterns = [
        r"openclaw-gateway",
        r"Genspark Claw\.app/Contents/MacOS/Genspark Claw$",
        r"acpx\b",
    ]
    return any(re.search(p, cmd, re.IGNORECASE) for p in patterns)


# ──────────────────────────────────────────────
# Local Collector
# ──────────────────────────────────────────────

async def _get_cwd(pid: int) -> str:
    """Get the cwd of a process. Works on macOS and Linux."""
    if platform.system() == "Linux":
        try:
            return os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            pass

    # macOS fallback: lsof
    try:
        proc = await asyncio.create_subprocess_exec(
            "lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        for line in stdout.decode().split("\n"):
            if line.startswith("n/"):
                return line[1:]
    except Exception:
        pass
    return ""


def _find_session_for_cwd(cwd: str) -> Optional[dict]:
    """Find the most recent Claude Code session for a working directory.

    Returns dict with keys: session_id, model, context_tokens, context_window,
    context_percent, first_user_message.
    """
    if not cwd or not CLAUDE_HOME.exists():
        return None

    # Claude Code encodes project path as directory name
    # e.g. /Users/foo/bar -> -Users-foo-bar
    encoded = cwd.replace("/", "-")
    project_dir = CLAUDE_HOME / "projects" / encoded

    if not project_dir.is_dir():
        return None

    # Find the active session (session_id file or most recent .jsonl)
    session_id = None
    session_id_file = project_dir / "session_id"
    if session_id_file.exists():
        session_id = session_id_file.read_text().strip()

    # Find session file
    session_file = None
    if session_id:
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            session_file = candidate

    if not session_file:
        # Fall back to most recently modified .jsonl
        jsonl_files = sorted(project_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if jsonl_files:
            session_file = jsonl_files[0]
            session_id = session_file.stem

    if not session_file:
        return None

    # Parse the session file — read from the end for efficiency
    model = ""
    total_input = 0
    first_user_msg = ""
    last_user_msg = ""

    try:
        with open(session_file) as f:
            lines = f.readlines()

        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") == "assistant" and not model:
                msg = entry.get("message", {})
                if isinstance(msg, dict):
                    model = msg.get("model", "")
                    usage = msg.get("usage", {})
                    if usage:
                        # Most recent usage gives us cumulative context size
                        total_input = (
                            usage.get("input_tokens", 0)
                            + usage.get("cache_creation_input_tokens", 0)
                            + usage.get("cache_read_input_tokens", 0)
                        )

            if entry.get("type") == "user" and not last_user_msg:
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            last_user_msg = c["text"][:500]
                            break
                elif isinstance(content, str):
                    last_user_msg = content[:500]

            if model and last_user_msg:
                break

        # Get first user message for context
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            first_user_msg = c["text"][:500]
                            break
                elif isinstance(content, str):
                    first_user_msg = content[:500]
                if first_user_msg:
                    break

    except Exception:
        return None

    context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
    context_percent = round(total_input / context_window * 100, 1) if context_window else None

    return {
        "session_id": session_id or "",
        "model": model,
        "context_tokens": total_input,
        "context_window": context_window,
        "context_percent": context_percent,
        "first_user_message": first_user_msg,
        "last_user_message": last_user_msg,
        "message_count": len(lines),
    }


async def collect_local(machine_id: str = "local") -> list[AgentNode]:
    """Collect all Claude Code / OpenClaw agents from local machine."""
    proc = await asyncio.create_subprocess_exec(
        "ps", "-eo", "pid=,ppid=,args=",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    all_procs: dict[int, dict] = {}
    agents: list[AgentNode] = []

    for line in stdout.decode().strip().split("\n"):
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if not m:
            continue
        pid, ppid, cmd = int(m.group(1)), int(m.group(2)), m.group(3)
        all_procs[pid] = {"pid": pid, "ppid": ppid, "command": cmd}

        agent_type = None
        if is_openclaw(cmd):
            agent_type = "openclaw"
        elif is_claude_code(cmd):
            agent_type = "claude_code"

        if agent_type:
            agents.append(AgentNode(
                id=f"{machine_id}-{pid}",
                type=agent_type,
                pid=pid,
                ppid=ppid,
                command=cmd[:300],
            ))

    # Fetch cwd in parallel
    cwd_tasks = {agent.pid: _get_cwd(agent.pid) for agent in agents}
    cwds = {}
    if cwd_tasks:
        results = await asyncio.gather(*cwd_tasks.values())
        cwds = dict(zip(cwd_tasks.keys(), results))

    for agent in agents:
        agent.working_dir = cwds.get(agent.pid, "")

        # Enrich Claude Code agents with session data
        if agent.type == "claude_code" and agent.working_dir:
            session = _find_session_for_cwd(agent.working_dir)
            if session:
                agent.session_id = session["session_id"]
                agent.model = session["model"]
                agent.context_tokens = session["context_tokens"]
                agent.context_window = session["context_window"]
                agent.context_percent = session["context_percent"]

    return _build_tree(agents, all_procs, machine_id)


def _build_tree(agents: list[AgentNode], all_procs: dict[int, dict], machine_id: str) -> list[AgentNode]:
    """Build parent-child tree by walking PPID chains."""
    agent_pids = {a.pid for a in agents}
    agent_by_pid = {a.pid: a for a in agents}
    children_set: set[int] = set()

    for agent in agents:
        current = agent.ppid
        visited: set[int] = set()
        while current > 1 and current not in visited:
            visited.add(current)
            if current in agent_pids:
                agent_by_pid[current].children.append(agent)
                children_set.add(agent.pid)
                break
            parent_proc = all_procs.get(current)
            if parent_proc:
                current = parent_proc["ppid"]
            else:
                break

    return [a for a in agents if a.pid not in children_set]


# ──────────────────────────────────────────────
# SSH Remote Collector
# ──────────────────────────────────────────────

# This script runs on the remote machine via `ssh host python3 -`
REMOTE_COLLECT_SCRIPT = r'''
import subprocess, json, os, re, platform, sys

CLAUDE_HOME = os.path.expanduser("~/.claude")
MODEL_CTX = {
    "claude-opus-4-6": 1000000, "claude-sonnet-4-6": 1000000,
    "claude-sonnet-4-5-20250514": 200000, "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000, "claude-haiku-4-5-20251001": 200000,
    "claude-3-opus": 200000,
}

def get_cwd(pid):
    if platform.system() == "Linux":
        try: return os.readlink(f"/proc/{pid}/cwd")
        except: pass
    try:
        r = subprocess.run(["lsof","-a","-d","cwd","-p",str(pid),"-Fn"],
                          capture_output=True, text=True, timeout=5)
        for l in r.stdout.split("\n"):
            if l.startswith("n/"): return l[1:]
    except: pass
    return ""

def get_session(cwd):
    if not cwd or not os.path.isdir(CLAUDE_HOME): return None
    encoded = cwd.replace("/", "-")
    pdir = os.path.join(CLAUDE_HOME, "projects", encoded)
    if not os.path.isdir(pdir): return None

    sid = None
    sf = os.path.join(pdir, "session_id")
    if os.path.exists(sf):
        with open(sf) as f: sid = f.read().strip()

    session_file = None
    if sid:
        c = os.path.join(pdir, f"{sid}.jsonl")
        if os.path.exists(c): session_file = c

    if not session_file:
        import glob
        files = sorted(glob.glob(os.path.join(pdir, "*.jsonl")),
                       key=os.path.getmtime, reverse=True)
        if files:
            session_file = files[0]
            sid = os.path.splitext(os.path.basename(files[0]))[0]

    if not session_file: return None

    model = ""; total_input = 0; first_msg = ""; last_msg = ""
    try:
        with open(session_file) as f: lines = f.readlines()
        for line in reversed(lines):
            try: e = json.loads(line)
            except: continue
            if e.get("type") == "assistant" and not model:
                msg = e.get("message", {})
                if isinstance(msg, dict):
                    model = msg.get("model", "")
                    u = msg.get("usage", {})
                    if u:
                        total_input = (
                            u.get("input_tokens", 0)
                            + u.get("cache_creation_input_tokens", 0)
                            + u.get("cache_read_input_tokens", 0)
                        )
            if e.get("type") == "user" and not last_msg:
                msg = e.get("message", {})
                c = msg.get("content","") if isinstance(msg,dict) else ""
                if isinstance(c, list):
                    for x in c:
                        if isinstance(x,dict) and x.get("type")=="text":
                            last_msg = x["text"][:500]; break
                elif isinstance(c, str): last_msg = c[:500]
            if model and last_msg: break

        for line in lines:
            try: e = json.loads(line)
            except: continue
            if e.get("type") == "user":
                msg = e.get("message", {})
                c = msg.get("content","") if isinstance(msg,dict) else ""
                if isinstance(c, list):
                    for x in c:
                        if isinstance(x,dict) and x.get("type")=="text":
                            first_msg = x["text"][:500]; break
                elif isinstance(c, str): first_msg = c[:500]
                if first_msg: break
    except: return None

    ctx_win = MODEL_CTX.get(model, 200000)
    return {"session_id": sid or "", "model": model, "context_tokens": total_input,
            "context_window": ctx_win,
            "context_percent": round(total_input/ctx_win*100,1) if ctx_win else None,
            "first_user_message": first_msg, "last_user_message": last_msg,
            "message_count": len(lines)}

def main():
    r = subprocess.run(["ps","-eo","pid=,ppid=,args="], capture_output=True, text=True)
    all_procs = {}; agents = []
    for line in r.stdout.strip().split("\n"):
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if not m: continue
        pid, ppid, cmd = int(m.group(1)), int(m.group(2)), m.group(3)
        all_procs[pid] = ppid
        if "grep" in cmd or "agent_monitor" in cmd: continue
        if re.search(r"--type=(gpu-process|renderer|utility|zygote)", cmd): continue
        if "Google Chrome" in cmd and "--user-data-dir" in cmd: continue
        is_cc = bool(re.search(r"(?:^|/)claude[\s$]|claude-agent-acp|claude-bridge|/\.claude/local/", cmd))
        is_oc = bool(re.search(r"openclaw-gateway|Genspark Claw.app/Contents/MacOS/Genspark Claw$|acpx\b", cmd, re.I))
        if is_cc or is_oc:
            cwd = get_cwd(pid)
            session = get_session(cwd) if is_cc else None
            agents.append({"pid":pid,"ppid":ppid,"type":"openclaw" if is_oc else "claude_code",
                          "command":cmd[:300],"working_dir":cwd,"session":session})

    print(json.dumps({"agents":agents,"all_procs":{str(k):v for k,v in all_procs.items()}}))

main()
'''


async def collect_ssh(host: str, ssh_args: list[str] | None = None,
                      machine_id: str = "") -> tuple[list[AgentNode], str]:
    """Collect agents from a remote machine via SSH.

    Returns (agents, error_message).
    """
    cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no"]
    if ssh_args:
        cmd.extend(ssh_args)
    cmd.extend([host, "python3", "-"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(
            input=REMOTE_COLLECT_SCRIPT.encode(),
            # timeout handled by ConnectTimeout in ssh
        )

        if proc.returncode != 0:
            err = stderr.decode()[:300]
            return [], f"SSH error (exit {proc.returncode}): {err}"

        data = json.loads(stdout.decode())
        all_procs_raw: dict[str, int] = data.get("all_procs", {})
        all_procs = {int(k): {"pid": int(k), "ppid": v, "command": ""} for k, v in all_procs_raw.items()}

        agents: list[AgentNode] = []
        for a in data.get("agents", []):
            node = AgentNode(
                id=f"{machine_id}-{a['pid']}",
                type=a["type"],
                pid=a["pid"],
                ppid=a["ppid"],
                command=a.get("command", ""),
                working_dir=a.get("working_dir", ""),
            )
            sess = a.get("session")
            if sess:
                node.session_id = sess.get("session_id", "")
                node.model = sess.get("model", "")
                node.context_tokens = sess.get("context_tokens")
                node.context_window = sess.get("context_window")
                node.context_percent = sess.get("context_percent")
            agents.append(node)

        return _build_tree(agents, all_procs, machine_id), ""

    except asyncio.TimeoutError:
        return [], "SSH connection timed out"
    except json.JSONDecodeError as e:
        return [], f"Failed to parse remote data: {e}"
    except Exception as e:
        return [], f"Collection failed: {e}"


# ──────────────────────────────────────────────
# Gemini Summarizer
# ──────────────────────────────────────────────

class Summarizer:
    def __init__(self, api_key: str = "", model: str = GEMINI_MODEL, proxy_url: str = ""):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.proxy_url = proxy_url  # Optional LiteLLM proxy
        self._cache: dict[str, tuple[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.proxy_url)

    async def summarize(self, agent: AgentNode) -> str:
        cache_key = f"{agent.pid}-{agent.working_dir}-{agent.session_id}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < SUMMARY_CACHE_TTL:
            return cached[0]

        if not self.enabled:
            return self._fallback(agent)

        context = self._build_context(agent)
        if not context.strip():
            return self._fallback(agent)

        prompt = (
            "Based on the following information about a running AI agent, "
            "write a concise 1-sentence summary (in the language of the user's messages, "
            "Chinese if messages are in Chinese) describing what task it's working on.\n\n"
            f"{context}\n\n"
            "Reply with ONLY the summary sentence."
        )

        try:
            summary = await self._call_llm(prompt)
            summary = summary.strip().strip('"').strip("'")
            self._cache[cache_key] = (summary, time.time())
            return summary
        except Exception:
            return self._fallback(agent)

    def _build_context(self, agent: AgentNode) -> str:
        parts = [
            f"Type: {agent.type}",
            f"Working directory: {agent.working_dir}",
        ]

        # Use session data for Claude Code
        session = None
        if agent.type == "claude_code" and agent.working_dir:
            session = _find_session_for_cwd(agent.working_dir)
        if session:
            if session.get("first_user_message"):
                parts.append(f"Initial user request: {session['first_user_message']}")
            if session.get("last_user_message"):
                parts.append(f"Most recent user message: {session['last_user_message']}")
            parts.append(f"Total messages in session: {session.get('message_count', '?')}")
        else:
            # Try git branch
            if agent.working_dir:
                try:
                    r = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=agent.working_dir, capture_output=True, text=True, timeout=3,
                    )
                    if r.returncode == 0:
                        parts.append(f"Git branch: {r.stdout.strip()}")
                except Exception:
                    pass

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        if self.proxy_url:
            return await self._call_openai_compat(prompt)
        return await self._call_gemini_direct(prompt)

    async def _call_gemini_direct(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.3},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Gemini API {resp.status}")
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai_compat(self, prompt: str) -> str:
        """Call via LiteLLM proxy or any OpenAI-compatible endpoint."""
        url = f"{self.proxy_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"LLM proxy {resp.status}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _fallback(self, agent: AgentNode) -> str:
        if agent.working_dir:
            return Path(agent.working_dir).name
        return ""


# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(title="Agent Monitor")

# Global state
_config: dict[str, Any] = {}
_summarizer: Summarizer = Summarizer()


async def _summarize_tree(agents: list[AgentNode]) -> None:
    """Recursively summarize all agents in the tree."""
    tasks = []
    for agent in agents:
        tasks.append(_summarize_single(agent))
    await asyncio.gather(*tasks)


async def _summarize_single(agent: AgentNode) -> None:
    agent.summary = await _summarizer.summarize(agent)
    if agent.children:
        await asyncio.gather(*[_summarize_single(c) for c in agent.children])


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/status")
async def get_status():
    machines: list[MachineStatus] = []

    # Collect from all machines in parallel
    async def _collect_local():
        m = MachineStatus(name="Local", host="local")
        try:
            m.agents = await collect_local("local")
            await _summarize_tree(m.agents)
        except Exception as e:
            m.status = "error"
            m.error = str(e)[:300]
        return m

    async def _collect_remote(cfg: dict):
        host = cfg["host"]
        name = cfg.get("name", host)
        mid = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        m = MachineStatus(name=name, host=host)
        try:
            agents, err = await collect_ssh(host, cfg.get("ssh_args"), mid)
            if err:
                m.status = "error"
                m.error = err
            m.agents = agents
            await _summarize_tree(agents)
        except Exception as e:
            m.status = "error"
            m.error = str(e)[:300]
        return m

    tasks = [_collect_local()]
    for remote in _config.get("machines", []):
        tasks.append(_collect_remote(remote))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            machines.append(MachineStatus(name="?", host="?", status="error", error=str(r)[:300]))
        else:
            machines.append(r)

    return {
        "machines": [m.to_dict() for m in machines],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "gemini_enabled": _summarizer.enabled,
        },
    }


PR_LINK_RE = re.compile(
    r"https://github\.com/[^\s/]+/[^\s/]+/pull/\d+"
    r"|(?:^|\s)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+#\d+)"
    r"|(?:^|\s)#(\d+)",
    re.MULTILINE,
)


def _extract_text(content: Any) -> str:
    """Extract plain text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result: {block.get('tool_use_id', '')[:8]}]")
                elif block.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content else ""


def _extract_pr_links(text: str) -> list[str]:
    """Find GitHub PR links in text."""
    links = []
    # Full URLs
    for m in re.finditer(
        r"https://github\.com/[^\s/]+/[^\s/]+/pull/\d+", text
    ):
        links.append(m.group(0))
    # owner/repo#123 format
    for m in re.finditer(r"(?:^|\s)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+#\d+)", text):
        ref = m.group(1)
        owner_repo, num = ref.rsplit("#", 1)
        links.append(f"https://github.com/{owner_repo}/pull/{num}")
    return list(dict.fromkeys(links))  # dedupe preserving order


def _read_session_messages(cwd: str) -> list[dict]:
    """Read session messages, returning user msgs + last assistant text."""
    if not cwd or not CLAUDE_HOME.exists():
        return []

    encoded = cwd.replace("/", "-")
    project_dir = CLAUDE_HOME / "projects" / encoded
    if not project_dir.is_dir():
        return []

    # Find active session
    session_id = None
    sid_file = project_dir / "session_id"
    if sid_file.exists():
        session_id = sid_file.read_text().strip()

    session_file = None
    if session_id:
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            session_file = candidate
    if not session_file:
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
        if jsonl_files:
            session_file = jsonl_files[0]

    if not session_file:
        return []

    messages: list[dict] = []
    try:
        with open(session_file) as f:
            lines = f.readlines()

        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = entry.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", "")
            text = _extract_text(content)
            if not text.strip():
                continue

            ts = entry.get("timestamp", "")
            pr_links = _extract_pr_links(text)

            # For assistant, extract only the text blocks (skip thinking/tool_use)
            if msg_type == "assistant" and isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts)
                if not text.strip():
                    continue

            messages.append({
                "type": msg_type,
                "text": text[:3000],  # cap size for API response
                "timestamp": ts,
                "pr_links": pr_links,
            })
    except Exception:
        return []

    return messages


@app.get("/api/session")
async def get_session_messages(cwd: str = ""):
    """Return conversation messages for a Claude Code session."""
    if not cwd:
        return {"messages": [], "error": "cwd required"}

    messages = _read_session_messages(cwd)
    return {"messages": messages, "total": len(messages)}


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def main():
    global _config, _summarizer

    parser = argparse.ArgumentParser(description="Agent Monitor Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 9900)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--gemini-model", type=str, default=GEMINI_MODEL, help="Gemini model name")
    parser.add_argument("--proxy-url", type=str, default=None, help="LiteLLM / OpenAI-compat proxy URL")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not config_path:
        default_cfg = Path(__file__).parent / "config.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            _config = yaml.safe_load(f) or {}

    _summarizer = Summarizer(
        api_key=args.gemini_key or "",
        model=args.gemini_model,
        proxy_url=args.proxy_url or "",
    )

    print(f"Agent Monitor  →  http://localhost:{args.port}")
    if _summarizer.enabled:
        print(f"  Gemini summarization: ON ({_summarizer.model})")
    else:
        print("  Gemini summarization: OFF (set --gemini-key or GEMINI_API_KEY)")
    if _config.get("machines"):
        print(f"  Remote machines: {', '.join(m.get('name', m['host']) for m in _config['machines'])}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
