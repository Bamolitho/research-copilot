.PHONY: test mypy lint lint-fix format format-check check run clean

SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts
PYTHON := uv run

# -------------------------
# Run agent
# -------------------------

run:
	$(PYTHON) python -m scripts.run_agent

# -------------------------
# Tests
# -------------------------

test:
	@if find $(TEST_DIR) -type f -name "test_*.py" | grep -q .; then \
		$(PYTHON) pytest $(TEST_DIR); \
	else \
		echo "No tests found."; \
	fi

mypy:
	$(PYTHON) mypy $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

# -------------------------
# Linting
# -------------------------

lint:
	$(PYTHON) ruff check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

lint-fix:
	$(PYTHON) ruff check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR) --fix

# -------------------------
# Formatting
# -------------------------

format:
	$(PYTHON) ruff format $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

format-check:
	$(PYTHON) ruff format --check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

# -------------------------
# Global checks
# -------------------------

check: lint mypy format-check test

# -------------------------
# Cleanup
# -------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
