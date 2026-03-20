.PHONY: setup pipeline dashboard train test lint clean matrix predict-aus predict-race predict-season championship backfill-weather migrate fetch-schedule live backfill backfill-season api serve export publish publish-round schedule

SEASON ?= 2026
ROUND ?= 1
STAGE ?= pre_weekend

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

pipeline:
	. .venv/bin/activate && python -m src.pipeline.run pipeline

pipeline-round:
	. .venv/bin/activate && python -m src.pipeline.run pipeline --round $(ROUND)

predict-race:
	. .venv/bin/activate && python -m src.pipeline.run predict --season $(SEASON) --round $(ROUND)

predict-season:
	. .venv/bin/activate && python -m src.pipeline.run predict-season --season $(SEASON)

championship:
	. .venv/bin/activate && python -m src.pipeline.run championship --season $(SEASON)

fetch-kaggle:
	. .venv/bin/activate && python scripts/seed_database.py

fetch-schedule:
	. .venv/bin/activate && python -m src.pipeline.run fetch-schedule --season $(SEASON)

train:
	. .venv/bin/activate && python -m src.pipeline.run train

dashboard:
	. .venv/bin/activate && streamlit run dashboard/app.py

test:
	. .venv/bin/activate && python -m pytest tests/ -v

lint:
	. .venv/bin/activate && ruff check src/ dashboard/ tests/ && ruff format src/ dashboard/ tests/

migrate:
	. .venv/bin/activate && python -m src.pipeline.run migrate

backfill-weather:
	. .venv/bin/activate && PYTHONPATH=. python scripts/backfill_weather.py

matrix:
	@. .venv/bin/activate && PYTHONPATH=. python -c "\
	import pandas as pd;\
	pd.set_option('display.max_columns',25);\
	pd.set_option('display.width',220);\
	pd.set_option('display.max_rows',25);\
	df=pd.read_parquet('data/processed/predictions/latest_matrix.parquet');\
	pos=[f'P{i}' for i in range(1,21)];\
	summary=['win_prob','podium_prob','top5_prob','expected_position'];\
	print('='*140);\
	print('FULL PROBABILITY MATRIX (drivers x finishing positions)');\
	print('='*140);\
	print();\
	print('POSITIONS P1-P10:');\
	print(df[[c for c in pos[:10] if c in df.columns]].to_string(float_format=lambda x:f'{x:6.1%}'));\
	print();\
	print('POSITIONS P11-P20:');\
	print(df[[c for c in pos[10:] if c in df.columns]].to_string(float_format=lambda x:f'{x:6.1%}'));\
	print();\
	print('SUMMARY:');\
	print(df[[c for c in summary if c in df.columns]].to_string(float_format=lambda x:f'{x:6.1%}' if x<1.5 else f'{x:6.1f}'));\
	"

live:
	. .venv/bin/activate && python -m src.pipeline.run live --season $(SEASON) --round $(ROUND)

backfill:
	. .venv/bin/activate && python -m src.pipeline.run backfill --season $(SEASON) --round $(ROUND)

backfill-season:
	. .venv/bin/activate && python -m src.pipeline.run backfill-season --season $(SEASON)

predict-aus:
	. .venv/bin/activate && PYTHONPATH=. python scripts/predict_australia_2026.py

api:
	. .venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve:
	@echo "Start 'make api' in one terminal, then 'ngrok http 8000' in another."
	@echo "Copy the ngrok URL and paste it into the frontend at f1.loukik.dev"

export:
	. .venv/bin/activate && python -m src.pipeline.run export --season $(SEASON)

publish:
	. .venv/bin/activate && python -m src.pipeline.run export --season $(SEASON) --push

publish-round:
	. .venv/bin/activate && python -m src.pipeline.run predict --season $(SEASON) --round $(ROUND) --stage $(STAGE) && \
	python -m src.pipeline.run export --season $(SEASON) --round $(ROUND) --push

schedule:
	. .venv/bin/activate && python -m src.pipeline.run schedule

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache dist build *.egg-info
