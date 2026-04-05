# Agent Monitor

Dashboard for monitoring OpenClaw and Claude Code instances across machines.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)

## Quick Start

```bash
git clone https://github.com/loveofcsdt/agent-monitor.git
cd agent-monitor
pip install -r requirements.txt
python monitor.py
```

Open **http://localhost:9900**

## What It Shows

- All running OpenClaw and Claude Code processes, with parent-child tree
- Claude Code context usage % (read from `~/.claude/` session data)
- Model name (e.g. `claude-opus-4-6`)
- Click any Claude Code to view conversation history in a side panel
  - User messages (first & last expanded, middle collapsed)
  - Last assistant response
  - GitHub PR links auto-extracted from messages

## Options

```bash
# Custom port
python monitor.py --port 8080

# Enable Gemini task summaries (1-sentence description per agent)
python monitor.py --gemini-key YOUR_GEMINI_API_KEY

# Use LiteLLM proxy instead of direct Gemini API
python monitor.py --proxy-url http://localhost:4000 --gemini-model gemini-2.5-flash
```

## Monitor Remote Machines

Create `config.yaml` (see `config.example.yaml`):

```yaml
machines:
  - name: "MacBook-Studio"
    host: "192.168.1.100"
    ssh_args: ["-i", "~/.ssh/id_rsa"]

  - name: "Dev Server"
    host: "dev-box"              # SSH config alias works
    ssh_args: ["-J", "jumphost"] # jump host, tailscale, etc.
```

```bash
python monitor.py --config config.yaml
```

Remote collection runs a self-contained Python script over SSH — no installation needed on the remote machine (just Python 3).

## Requirements

- Python 3.11+
- `pip install -r requirements.txt` (fastapi, uvicorn, aiohttp, pyyaml)
- macOS or Linux (uses `ps` and `lsof` for process detection)
