# Dockerfile for https://github.com/kapoorlab/NapaTrackMater
# Author: NicolasCARPi https://github.com/NicolasCARPi/

# start with the debian version, not alpine as it causes many issues
FROM python:3.7-slim-buster

# create a user instead of using root user
RUN useradd -mls /bin/bash worker

# this is required by dependencies
RUN apt-get update \
    && apt-get install -y libglib2.0-dev libgl1-mesa-glx python3-fontconfig \
    && rm -rf /var/lib/apt/lists/*

# set some python/pip env vars, adjust PATH
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  PYTHONPATH=/home/worker/lib/python \
  PATH=/home/worker/.local/bin:${PATH} \
  DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app
RUN chown worker:worker .
USER worker

COPY requirements.txt setup.py README.md .

RUN pip install -vv --no-cache-dir --prefer-binary --user -r requirements.txt
# chown is necessary or everything will be root owned
COPY --chown=worker:worker . .
# install the package
RUN python setup.py develop --user
# default command just shows the help
CMD ["track", "-h"]
