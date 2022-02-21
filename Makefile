DUMMY: lint test

lint:
	flake8 src test
	mypy src
	pydocstyle src
format:
	black .
test:
	pytest
