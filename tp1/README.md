# PSTAL – TP1 : étiqueteur POS avec réseau récurrent (GRU)

Étiquetage en parties du discours (UPOS) du corpus français **Sequoia**,
avec un réseau Embedding → GRU → Linear en PyTorch.

## Fichiers

| Fichier | Rôle |
|---|---|
| `model.py` | Le modèle `GRUTagger` (partagé par les deux scripts) |
| `train_postag.py` | Lecture du corpus, vocabulaires, padding, entraînement, sauvegarde du meilleur modèle |
| `predict_postag.py` | Chargement du modèle et prédiction des POS phrase par phrase |
| `REVISION_ORAL.md` | Fiche de révision : explications, résultats, questions probables |
| `resultats/train.log` | Sortie complète de l'entraînement (loss et accuracy par epoch) |
| `resultats/postag.pt` | Modèle entraîné (paramètres + vocabulaires + hyper-paramètres) |
| `resultats/test.pred.conllu` | Prédictions sur le corpus de test |
| `resultats/eval_test.txt` | Sortie de `evaluate.py` sur le test |

## Prérequis

- Python 3.11, `torch`, `conllu`, `numpy`
- Le dépôt du cours `pstal-etu` cloné **à côté** de ce dossier
  (les scripts utilisent `../../pstal-etu/lib/conllulib.py` et le corpus `../../pstal-etu/sequoia/`)

## Utilisation

```bash
SEQ=../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple

# Entraînement (options : --epochs, --max-len, --d-embed, --d-hidden, --seed ; voir --help)
python train_postag.py --train $SEQ.train --dev $SEQ.dev --model postag.pt

# Prédiction
python predict_postag.py --model postag.pt --input $SEQ.test --output test.pred.conllu

# Évaluation
python ../../pstal-etu/lib/evaluate.py -p test.pred.conllu -g $SEQ.test -t $SEQ.train
```

## Résultats (seed 42, 20 epochs, d_embed = d_hidden = 100, L = 40)

| Corpus | Accuracy UPOS | Accuracy sur les mots inconnus (OOV) |
|---|---|---|
| test | 90,43 % | 51,75 % |

Le modèle conservé est celui de l'epoch où la loss sur le dev est minimale.
La principale source d'erreurs est constituée des mots inconnus : le vecteur `<UNK>`
n'est jamais vu pendant l'entraînement.
