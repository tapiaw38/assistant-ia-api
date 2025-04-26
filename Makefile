API_SERVICE=assistant-ia-api

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

test:
	docker-compose exec $(API_SERVICE) pytest

test-v:
	docker-compose exec $(API_SERVICE) pytest -v