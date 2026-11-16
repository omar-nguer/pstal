import sys
sys.path.append("../../pstal-etu/lib")
import argparse
import torch
from conllulib import CoNLLUReader, Util
from model import GRUTagger

# --- 0. Arguments de la ligne de commande ---
parser = argparse.ArgumentParser(description="Prédit les POS d'un corpus avec un modèle entraîné.")
parser.add_argument("--model", required=True, help="modèle entraîné (.pt)")
parser.add_argument("--input", required=True, help="corpus à étiqueter (CoNLL-U)")
parser.add_argument("--output", required=True, help="fichier de sortie avec les POS prédites")
args = parser.parse_args()

# --- 1. Recréer le modèle et charger les paramètres appris ---
checkpoint = torch.load(args.model, weights_only=False)
vocab = checkpoint["vocab"]
model = GRUTagger(**checkpoint["hyper"])        # modèle vide, de la bonne taille
model.load_state_dict(checkpoint["model"])      # on y met les paramètres appris
model.eval()                                    # mode prédiction : pas de dropout
rev_upos = Util.rev_vocab(vocab["upos"])        # numéro -> étiquette (2 -> "NOUN")

# --- 2. Prédire phrase par phrase et écrire le résultat ---
with open(args.input, encoding="UTF-8") as fin, open(args.output, "w", encoding="UTF-8") as fout:
    for sent in CoNLLUReader(fin).readConllu():
        ids = CoNLLUReader.to_int_from_vocab_sent(sent, ["form"], "<UNK>", vocab)["form"]
        x = torch.LongTensor([ids])             # (1, n) : un "batch" d'une seule phrase
        with torch.no_grad():
            logits = model(x)                   # (1, n, 17)
        t_hat = logits.argmax(dim=2)[0]         # (n,) : une étiquette par mot
        for tok, tag_id in zip(sent, t_hat.tolist()):
            tok["upos"] = rev_upos[tag_id]      # on remplace la vraie étiquette par la prédiction
        fout.write(sent.serialize())            # réécrit la phrase au format CoNLL-U

print("Prédictions écrites dans", args.output)