import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, encoded_image_size):
        super(RNNDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_input_size = embed_size + encoded_image_size
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        sequence_length = embeddings.size(1)
        features = features.unsqueeze(1).repeat(1, sequence_length, 1)
        embeddings = torch.cat((features, embeddings), dim=2)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(self.dropout(hiddens))
        return outputs
