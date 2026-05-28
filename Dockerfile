FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Telecharger model.pkl depuis HF au moment du build
RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Adja26/sensante', filename='models/model.pkl', repo_type='space', local_dir='.')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]