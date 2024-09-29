FROM python:3.10-slim

WORKDIR  /app

RUN pip install poetry

COPY pyproject.toml /app

RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

COPY . /app

EXPOSE 5000

CMD ["python3", "main.py"]