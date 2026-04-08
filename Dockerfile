FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pickle-persist dir for per-user platform state
RUN mkdir -p /app/data-persist

CMD ["python", "main.py"]
