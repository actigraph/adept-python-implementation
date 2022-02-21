DUMMY: lint test

lint:
	flake8 src tests
	mypy src
	pydocstyle src
format:
	black .
test:
	python -m pytest
