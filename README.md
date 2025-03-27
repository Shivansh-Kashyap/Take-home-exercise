# Multi-Task Sentence Transformer (Dockerized)

This project demonstrates a multi-task NLP model using a shared SentenceTransformer encoder (`paraphrase-MiniLM-L6-v2`) with two task-specific heads for:

- Task A: Topic Classification
- Task B: Sentiment Analysis

## Usage

```bash
docker build -t multi-task-model .
docker run --rm multi-task-model
```

The container will simulate training on synthetic data and print loss and accuracy metrics for each task.

## Requirements

- Python 3.9+
- Docker
