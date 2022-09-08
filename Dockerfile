FROM python:3.10

WORKDIR /usr/local/foreground

COPY pyproject.toml .
COPY foreground foreground

RUN pip install .

RUN python foreground/download.py

COPY foreground/app.py app.py

ENTRYPOINT ["uvicorn", "app:app", "--reload"]