FROM python:3.12-slim

# libgomp install
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
