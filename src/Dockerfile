FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the search models once at build time so the app starts faster
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false"]