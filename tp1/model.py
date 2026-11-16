import torch
from torch import nn

class GRUTagger(nn.Module):
    """Étiqueteur POS : Embedding -> GRU -> Linear, une prédiction par mot."""

    def __init__(self, d_embed, d_hidden, d_in, d_out):
        super().__init__()
        self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)   # d_in = taille vocab mots
        self.gru = nn.GRU(d_embed, d_hidden, batch_first=True, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.decision = nn.Linear(d_hidden, d_out)                # d_out = nb d'étiquettes

    def forward(self, idx_words):
        embedded = self.embed(idx_words)        # (B, L)          -> (B, L, d_embed)
        hidden = self.gru(embedded)[0]          # (B, L, d_embed) -> (B, L, d_hidden). ""[0] : son état à chaque mot, de forme (16, 40, 100) ;[1] : seulement son état après le dernier mot, une sorte de résumé de la phrase, de forme (1, 16, 100).""
        return self.decision(self.dropout(hidden))  # -> (B, L, d_out)