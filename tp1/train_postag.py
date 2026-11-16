import sys
sys.path.append("../../pstal-etu/lib")
import argparse
import copy
import torch
from torch import nn
from conllulib import CoNLLUReader, Util
from model import GRUTagger

PAD_ID = 0      # numéro du padding : fixe, ce n'est pas un réglage

# --- 0. Arguments de la ligne de commande ---
parser = argparse.ArgumentParser(description="Entraîne un étiqueteur POS (Embedding + GRU).")
parser.add_argument("--train", required=True, help="corpus d'entraînement (CoNLL-U)")
parser.add_argument("--dev", required=True, help="corpus de développement (CoNLL-U)")
parser.add_argument("--model", default="postag.pt", help="fichier où sauvegarder le modèle")
parser.add_argument("--epochs", type=int, default=20, help="nombre d'epochs")
parser.add_argument("--max-len", type=int, default=40, help="longueur L des phrases (padding/troncage)")
parser.add_argument("--d-embed", type=int, default=100, help="taille des embeddings")
parser.add_argument("--d-hidden", type=int, default=100, help="taille de l'état caché du GRU")
parser.add_argument("--seed", type=int, default=42, help="graine du hasard (reproductibilité)")  ##on fixe les argument pris au hazard au debut pour avoir les meme resultats entre deux entrainements
args = parser.parse_args()
Util.init_seed(args.seed)

def pad_tensor(int_lists, max_len):
    """Transforme une liste de phrases (listes d'entiers de longueurs variables)
    en un tenseur de taille (nb_phrases, max_len), tronqué et complété par PAD_ID."""
    tensor = torch.full((len(int_lists), max_len), PAD_ID, dtype=torch.long)
    for i, sent in enumerate(int_lists):
        sent = sent[:max_len]                       # troncage
        tensor[i, :len(sent)] = torch.tensor(sent)  # le reste reste à PAD_ID
    return tensor


def perf(model, dev_loader, criterion):
    """Calcule la loss moyenne et l'accuracy (sans le padding) sur le dev."""
    model.eval()                                    # désactive le dropout
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():                           # pas de gradients sur le dev
        for x, y in dev_loader:
            logits = model(x)                       # (B, L, nb_étiquettes)
            total_loss += criterion(logits.transpose(1, 2), y).item()
            t_hat = logits.argmax(dim=2)            # (B, L) étiquettes prédites
            mask = (y != PAD_ID)                    # on ignore le padding
            correct += ((t_hat == y) * mask).sum().item()
            total += mask.sum().item()
    return total_loss / len(dev_loader), correct / total


def fit(model, train_loader, dev_loader, epochs):
    """Entraîne le modèle et garde les paramètres de la meilleure epoch (dev_loss minimale)."""
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters())
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()                               # active le dropout
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()                   # (a) remise à zéro des gradients
            logits = model(x)                       # (b) prédiction
            loss = criterion(logits.transpose(1, 2), y)   # calcul de l'erreur
            loss.backward()                         # (c) calcul des gradients
            optimizer.step()                        #     mise à jour des paramètres
            total_loss += loss.item()
        dev_loss, dev_acc = perf(model, dev_loader, criterion)
        print(f"Epoch {epoch+1} : train_loss = {total_loss/len(train_loader):.4f} "
              f"| dev_loss = {dev_loss:.4f} | dev_acc = {dev_acc:.4f}")
        if dev_loss < best_loss:                    # meilleure epoch jusqu'ici ?
            best_loss, best_epoch = dev_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)               # on garde le meilleur modèle
    print(f"Meilleur modèle : epoch {best_epoch} (dev_loss = {best_loss:.4f})")


# --- 1. Train : on construit les vocabulaires en même temps ---
with open(args.train, encoding="UTF-8") as f:
    ints_train, vocab = CoNLLUReader(f).to_int_and_vocab({"form": ["<PAD>", "<UNK>"], "upos": ["<PAD>"]})

# --- 2. Dev : on réutilise les vocabulaires du train (mots inconnus -> <UNK>) ---
with open(args.dev, encoding="UTF-8") as f:
    ints_dev = CoNLLUReader(f).to_int_from_vocab(["form", "upos"], "<UNK>", vocab)

# --- 3. Padding + regroupement en batches ---
x_train, y_train = pad_tensor(ints_train["form"], args.max_len), pad_tensor(ints_train["upos"], args.max_len)
x_dev,   y_dev   = pad_tensor(ints_dev["form"], args.max_len),   pad_tensor(ints_dev["upos"], args.max_len)
train_loader = Util.dataloader([x_train], [y_train], batch_size=16, shuffle=True)
dev_loader   = Util.dataloader([x_dev],   [y_dev],   batch_size=16, shuffle=False)

# --- 4. Création et entraînement du modèle ---
model = GRUTagger(args.d_embed, args.d_hidden, len(vocab["form"]), len(vocab["upos"]))
fit(model, train_loader, dev_loader, epochs=args.epochs)

# --- 5. Sauvegarde : paramètres + vocabulaires + hyper-paramètres ---
torch.save({
    "model": model.state_dict(),
    "vocab": vocab,
    "hyper": {"d_embed": args.d_embed, "d_hidden": args.d_hidden,
              "d_in": len(vocab["form"]), "d_out": len(vocab["upos"])},
}, args.model)
print("Modèle sauvegardé dans", args.model)