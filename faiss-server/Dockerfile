FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-b", ":8080", "faiss_app:app"]
