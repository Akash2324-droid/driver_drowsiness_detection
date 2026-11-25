FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 10000

CMD ["sh", "-c", "streamlit run frontend/app.py --server.address 0.0.0.0 --server.port ${PORT:-10000}"]
