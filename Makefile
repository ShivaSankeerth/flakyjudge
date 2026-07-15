.PHONY: test lint e1 e2 e3 e4 figures

test:
	uv run pytest -q

lint:
	uv run ruff check src tests experiments

e1:
	uv run python experiments/e1_anchor.py

e2:
	uv run python experiments/e2_noise_floor.py

e3:
	uv run python experiments/e3_paraphrase.py

e4:
	uv run python experiments/e4_verbosity.py

figures:
	uv run python analysis/make_figures.py
