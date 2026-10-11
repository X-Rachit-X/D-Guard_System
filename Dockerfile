FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY notebooks/kaggle_train_dguard.py ./notebooks/
COPY runs/ ./runs/

# Not in the image (licence and size): mount them at runtime
#   -v /path/to/milk10k:/app/milk10k
#   -v /path/to/outputs:/app/outputs      (calibration bundle, retrieval index)
#   and put the trained weights at runs/milk10k_resnet50_seed42/checkpoints/best.pt

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/dguard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
