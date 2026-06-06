FROM python:3.14


WORKDIR /code

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir -r /code/requirements.txt


COPY . /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
