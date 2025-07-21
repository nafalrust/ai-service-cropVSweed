FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir virtualenv \
    && virtualenv venv \
    && . venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY . .

EXPOSE 5000

CMD ["/bin/bash", "-c", ". venv/bin/activate && python app.py --host=0.0.0.0"]
