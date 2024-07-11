FROM nvcr.io/nvidia/pytorch:24.06-py3

RUN apt-get update && apt-get install -y git

WORKDIR /app/

COPY requirements/requirements.txt /app/requirements/
RUN pip install -r /app/requirements/requirements.txt
RUN mkdir /app/data
RUN mkdir /app/data/raw

# copy code and models
#ADD models /app/models
#ADD data /app/data
ADD src /app/src
ADD scripts /app/scripts
WORKDIR /app

COPY .env /app/.env

ENV PYTHONPATH="${PYTHONPATH}:/app/src/generativezoo"