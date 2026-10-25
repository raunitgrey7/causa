# causa developer entry points. Every target goes through uv, so the interpreter and the lock file are the
# ones the repository pins.
UV ?= uv
PORT ?= 8773

.PHONY: setup lint format test bench bench-quick train serve clean docker-build docker-run

setup:
	$(UV) sync --all-groups
	$(UV) run causa generate experiment --n 40000 --lift 0.05 --out data/experiment.csv
	$(UV) run causa generate campaign --n 20000 --out data/campaign.csv

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest -q

# The published run: writes benchmarks/results.json and results.md, copies the tables into README.md and
# fails if any check in benchmarks/thresholds.json fails. Three to five minutes on two worker processes,
# depending on what else the machine is running.
bench:
	$(UV) run python -m causa.bench --check --update-readme

# Seconds-long smoke run with tiny sizes; writes to build/ so the published results stay untouched.
bench-quick:
	$(UV) run python -m causa.bench --quick --out build/bench-quick

train:
	$(UV) run causa train

serve:
	$(UV) run causa serve --port $(PORT) --train-if-missing

clean:
	rm -rf build .pytest_cache .ruff_cache .hypothesis data/*.csv data/*.sqlite artifacts/*.joblib
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build:
	docker build -t causa:latest .

docker-run:
	docker run --rm -p $(PORT):8773 causa:latest
