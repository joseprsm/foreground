FROM python:3.10

WORKDIR /usr/local/foreground

COPY pyproject.toml .
COPY foreground foreground

RUN pip install numpy==1.23.1 scikit-image>=0.15.0

RUN pip install .

RUN python foreground/download.py

ENTRYPOINT ["uvicorn", "foreground.app:app", "--reload", "--host", "0.0.0.0"]