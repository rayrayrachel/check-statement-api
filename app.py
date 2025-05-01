from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
EMBEDDING_DIM = 110
HIDDEN_DIM = 256
MAX_LEN = 50 

# Load vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden_cat)
        output = self.fc(dropped)
        return self.sigmoid(output)

model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
model.load_state_dict(torch.load('lstm_cv_model.pth', map_location=device))
model.to(device)
model.eval()

def predict_statement(statement, max_len=MAX_LEN):
    tokens = word_tokenize(statement.lower())
    input_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(input_ids) < max_len:
        input_ids += [vocab['<PAD>']] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze().item()

    return output

# AIP
class TextRequest(BaseModel):
    paragraph: str

@app.post("/predict")
def predict_paragraph(req: TextRequest):
    sentences = sent_tokenize(req.paragraph)

    good_sentences = []
    bad_sentences = []
    results = []

    for sent in sentences:
        score = predict_statement(sent)
        label = "Good" if score >= 0.5 else "Bad"

        if label == "Good":
            good_sentences.append({
                "sentence": sent,
                "score": round(score, 4)
            })
        else:
            bad_sentences.append({
                "sentence": sent,
                "score": round(score, 4)
            })

        results.append({
            "sentence": sent,
            "score": round(score, 4),
            "predicted_label": label
        })

    good_percentage = (len(good_sentences) / len(sentences)) * 100 if sentences else 0

    return {
        "good_sentences": good_sentences,
        "bad_sentences": bad_sentences,
        "good_percentage": round(good_percentage, 2),
        "all_results": results
    }
