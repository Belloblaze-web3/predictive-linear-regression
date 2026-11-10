FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-streamlit.txt pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY app.py ./app.py

RUN pip install --upgrade pip && pip install -r requirements-streamlit.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
