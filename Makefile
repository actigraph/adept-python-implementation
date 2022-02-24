DUMMY: lint test

lint:
	flake8 pyadept tests
	mypy pyadept
	pydocstyle pyadept
format:
	black .
test:
	python -m pytest
