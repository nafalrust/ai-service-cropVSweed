FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_entire.pt .
COPY model2.pt .
COPY label_encoder.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
EXPOSE 5000

CMD ["/bin/bash", "-c", ". venv/bin/activate && python app.py --host=0.0.0.0"]
