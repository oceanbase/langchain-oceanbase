.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/


# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

# comprehensive tests run the unified test suite
comprehensive_test comprehensive_tests:
	poetry run python tests/test_comprehensive.py

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/oceanbase --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_oceanbase
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_oceanbase -name '*.py')
	poetry run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'integration_test             - run integration tests'
	@echo 'comprehensive_test           - run comprehensive tests (CI, compatibility, integration)'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'


# ---- onboarding additions start ----
.PHONY: help docker-up docker-up-seek docker-down docker-down-seek docker-logs docker-logs-seek \
        fmt lint typecheck test

help:
	@echo "Usage:"
	@echo "  make docker-up          Start OceanBase (docker-compose.yml)"
	@echo "  make docker-up-seek     Start SeekDB (docker-compose.seekdb.yml)"
	@echo "  make docker-down        Stop OceanBase and remove volumes"
	@echo "  make docker-down-seek   Stop SeekDB and remove volumes"
	@echo "  make docker-logs        Follow OceanBase logs"
	@echo "  make docker-logs-seek   Follow SeekDB logs"
	@echo "  make fmt                Format code with black"
	@echo "  make lint               Run ruff linter"
	@echo "  make typecheck          Run mypy type checks"
	@echo "  make test               Run pytest"

# Docker-compose management
docker-up:
	docker-compose -f docker-compose.yml up -d

docker-up-seek:
	docker-compose -f docker-compose.seekdb.yml up -d

docker-down:
	docker-compose -f docker-compose.yml down -v

docker-down-seek:
	docker-compose -f docker-compose.seekdb.yml down -v

docker-logs:
	docker-compose -f docker-compose.yml logs -f

docker-logs-seek:
	docker-compose -f docker-compose.seekdb.yml logs -f

# Development helpers (only active if these tools are in your toolchain)
fmt:
	black .

lint:
	ruff check .

typecheck:
	mypy .

test:
	pytest -q
# ---- onboarding additions end ----