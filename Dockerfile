# OpenStethoscope — self-contained public build
#
# Build:
#   docker build --build-arg HF_TOKEN=<your_huggingface_token> -t openstetho .
#
# Run:
#   docker run -p 8080:8080 openstetho
#
# HuggingFace token: create a free account at huggingface.co,
# accept the HeAR model license at huggingface.co/google/hear,
# then generate a token at huggingface.co/settings/tokens.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download HeAR model from HuggingFace (baked into image)
COPY download_hear.py ./
ARG HF_TOKEN
RUN HF_TOKEN=$HF_TOKEN python download_hear.py

# App code and models
COPY app/server.py app/index.html ./
COPY models/ /app/models/

ENV MODELS_DIR="/app/models"
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
