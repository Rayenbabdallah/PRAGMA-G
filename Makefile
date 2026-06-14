.PHONY: test lint train serve deploy

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

train:
	python -m src.training.pretrain --config configs/pragma_s.yaml
	python -m src.training.finetune --config configs/pragma_s.yaml

serve:
	docker-compose up

deploy:
	docker-compose build api
