FROM python:3.12 as builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_HOME="/opt/poetry"
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -
RUN poetry config virtualenvs.create false
WORKDIR /app
COPY pyproject.toml /app/
COPY poetry.lock /app/
RUN poetry install --no-interaction --without dev --no-ansi --no-root -vvv

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="UTF-8"
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/
CMD ["streamlit", "run", "home.py", "--server.port", "8080"]