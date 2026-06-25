FROM python:3.11-slim-buster
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Run embeddings first, then start app
CMD ["sh", "-c", "python store_index.py && python app.py"]
