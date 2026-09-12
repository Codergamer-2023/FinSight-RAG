FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --upgrade \
    pip

RUN python -m pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]