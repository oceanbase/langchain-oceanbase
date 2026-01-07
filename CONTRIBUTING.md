# Contributing

Thank you for contributing! This document explains how to get a local development environment running, run tests, and submit a PR.

## Prerequisites

- Git
- Python 3.10+ (recommended)
- Poetry (or pip + virtualenv)
- Docker & Docker Compose
- (Optional) OpenAI API key if you plan to run examples that generate embeddings or use an LLM

## Development Setup

1. Clone the repo:
```bash
git clone https://github.com/oceanbase/langchain-oceanbase.git
cd langchain-oceanbase
```

2. Create a virtual environment and install dependencies:
- With Poetry:
```bash
poetry install
poetry shell
```
- With pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Bring up the local database for development (OceanBase):
```bash
make docker-up
# or for SeekDB lightweight alternative:
make docker-up-seek
```

4. Environment variables (examples)
Create a `.env` file or export environment variables:
```
OB_HOST=127.0.0.1
OB_PORT=3306
OB_USER=root
OB_PASSWORD=yourpassword
OB_DB=langchain_ob_demo
OPENAI_API_KEY=sk-xxxx
```

## Running tests

- Unit tests:
```bash
pytest tests/unit
```

- Integration tests (requires docker-compose services up):
```bash
make docker-up
pytest tests/integration
```

- Linting / type checks:
```bash
ruff check .
mypy .
```

## Code style

- Formatting: black
- Linting: ruff
- Type checks: mypy
- Commit messages: follow Conventional Commits or keep concise, descriptive messages.

## Submitting PRs

1. Create a branch with a descriptive name:
```bash
git checkout -b feat/onboard-docker-compose
```

2. Make edits and add tests where applicable.

3. Run tests and linters locally.

4. Commit changes:
```bash
git add .
git commit -m "feat: add docker-compose and onboarding docs"
git push origin feat/onboard-docker-compose
```

5. Open a PR describing the change. Include:
- Problem you're solving
- Any migration or manual steps
- Testing steps

## PR review checklist

- [ ] Tests added/updated
- [ ] Linting passes
- [ ] Clear description & motivation
- [ ] Updated docs if needed

## Makefile targets

- `make docker-up` — start OceanBase
- `make docker-up-seek` — start SeekDB (alternative)
- `make docker-down` — stop OceanBase and remove volumes
- `make docker-logs` — follow OceanBase logs

Thank you for contributing!