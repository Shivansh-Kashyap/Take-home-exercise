from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiTaskSentenceModel(nn.Module):
    def __init__(self, transformer_model="paraphrase-MiniLM-L6-v2", hidden_size=384, num_labels_task_a=3, num_labels_task_b=2):
        super(MultiTaskSentenceModel, self).__init__()
        self.encoder = SentenceTransformer(transformer_model)
        self.task_a_head = nn.Linear(hidden_size, num_labels_task_a)
        self.task_b_head = nn.Linear(hidden_size, num_labels_task_b)
        self.task_a_labels = ["Finance", "Entertainment", "Technology"]
        self.task_b_labels = ["Negative", "Positive"]

    def forward(self, sentences):
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        logits_a = self.task_a_head(embeddings)
        logits_b = self.task_b_head(embeddings)
        return logits_a, logits_b

    def predict(self, sentences):
        logits_a, logits_b = self.forward(sentences)
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        pred_a = torch.argmax(probs_a, dim=1)
        pred_b = torch.argmax(probs_b, dim=1)
        for i, sentence in enumerate(sentences):
            print(f"Sentence: {sentence}")
            print(f"\tPredicted Topic: {self.task_a_labels[pred_a[i]]}")
            print(f"\tPredicted Sentiment: {self.task_b_labels[pred_b[i]]}\n")


model = MultiTaskSentenceModel()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()


sentences = [
    "The stock market crashed due to inflation.",
    "The film was a cinematic masterpiece.",
    "Tech companies are investing in AI.",
    "The weather ruined my mood today.",
    "Quantum computing is the next big thing."
]

task_a_labels = torch.tensor([0, 1, 2, 1, 2])
task_b_labels = torch.tensor([0, 1, 1, 0, 1])


model.train()
for epoch in range(3):
    logits_a, logits_b = model(sentences)
    loss_a = loss_fn(logits_a, task_a_labels)
    loss_b = loss_fn(logits_b, task_b_labels)
    total_loss = loss_a + loss_b

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    pred_a = torch.argmax(logits_a, dim=1)
    pred_b = torch.argmax(logits_b, dim=1)
    acc_a = (pred_a == task_a_labels).float().mean().item()
    acc_b = (pred_b == task_b_labels).float().mean().item()

    print(f"Epoch {epoch+1} | Loss: {total_loss.item():.4f} | Task A Acc: {acc_a:.2f} | Task B Acc: {acc_b:.2f}")
