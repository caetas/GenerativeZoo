FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN apt-get update && apt-get install -y git

COPY requirements/requirements_docker.txt /app/requirements/
RUN pip install -r /app/requirements/requirements_docker.txt
RUN mkdir /app/data
RUN mkdir /app/src
RUN mkdir /app/models

WORKDIR /app/

ENV PYTHONPATH="${PYTHONPATH}:/app/src/generativezoo"