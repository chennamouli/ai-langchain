# ai-langchain

LangChain practice repository


## Setup & Local development (macOS / zsh)

Follow these consolidated steps to create a local development workspace, manage a `.venv`, install dependencies, and run example agents.

1) Create and activate a virtual environment:

```bash
cd /Users/cgajam/workspace/ai-langchain
python3 -m venv .venv
source .venv/bin/activate
```

2) Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure environment variables:

```bash
cp .env.template .env
# Edit `.env` and set your `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (do NOT commit `.env`).
```

4) Run agents:

**Claude agent example:**
```bash
python agents/claude_agent.py
```

**OpenAI agent example:**
```bash
python agents/agent_template.py
```

When adding packages during development:

1. Install the package into the active virtualenv:

```bash
pip install <package-name>
```

2. Capture installed versions to `requirements.txt`:

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore: update requirements after adding <package-name>"
```

Best practices:

- Keep `.venv/` and `.env` in `.gitignore` (this repo already has `.gitignore`).
- Prefer `pip freeze > requirements.txt` for small projects. For reproducible workflows consider `pip-tools` (`pip-compile`) or Poetry.
- Review `requirements.txt` diffs before committing to avoid accidental upgrades.
- In CI or containers, provide `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` via secure environment variables or secret manager — never commit secrets.


