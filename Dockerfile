FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["./start.sh"]
