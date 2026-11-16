# TP1 – Fiche de révision pour l'oral

Étiqueteur POS (parties du discours) avec un réseau récurrent GRU, corpus Sequoia.

---

## 1. Le TP en 30 secondes

> On entraîne un réseau de neurones à donner, pour **chaque mot** d'une phrase, sa catégorie grammaticale (NOUN, VERB, DET…).
> Les mots sont transformés en numéros, puis en vecteurs (**embedding**). Un **GRU** lit la phrase pour tenir compte du contexte, et une **couche linéaire** donne un score pour chacune des 17 étiquettes.
> On l'entraîne sur le `.train`, on choisit le meilleur modèle grâce au `.dev`, et on donne le score final sur le `.test` : **90,43 %**.

```
L'    été   arrive  à    Marseille  .
DET   NOUN  VERB    ADP  PROPN      PUNCT
```

C'est de l'**étiquetage de séquences** : la sortie a **la même longueur** que l'entrée (une étiquette par mot).

---

## 2. Le parcours d'un batch dans le réseau (à savoir refaire au tableau)

```
x : (B, L) = (16, 40)            numéros des mots (avec padding)
   │  nn.Embedding(8289, 100)    chaque numéro → un vecteur de 100 nombres appris
   ▼
(16, 40, 100)
   │  nn.GRU(100, 100)           lit la phrase de gauche à droite, garde une mémoire du contexte
   ▼
(16, 40, 100)                    on prend gru(...)[0] : l'état à CHAQUE mot
   │  Dropout(0.1) + nn.Linear(100, 17)
   ▼
(16, 40, 17)                     logits : 17 scores par mot
   │  argmax(dim=2)
   ▼
(16, 40)                         étiquette prédite pour chaque mot
```

B = taille du batch, L = longueur des phrases, 17 = 16 étiquettes UPOS + `<PAD>`.

---

## 3. Les données

### Format CoNLL-U
- Un mot par ligne, une ligne vide entre deux phrases, 13 colonnes séparées par des tabulations. Les lignes `#` sont des métadonnées.
- **Entrée** : colonne 2, `FORM` (le mot). **Sortie à prédire** : colonne 4, `UPOS`.

### Train / dev / test
| Fichier | Rôle |
|---|---|
| `.train` (2 184 phrases) | apprendre les paramètres |
| `.dev` (402 phrases) | faire ses choix : meilleure epoch, réglages |
| `.test` | **une seule fois à la fin**, pour un score honnête |

**Pourquoi ne pas régler le modèle sur le test ?** Le score deviendrait trop optimiste : il ne mesurerait plus la performance sur de vraies nouvelles phrases.

### Vocabulaires (`to_int_and_vocab` / `to_int_from_vocab`)
- Chaque mot et chaque étiquette reçoit un numéro, grâce à un `defaultdict` : un mot nouveau prend comme numéro la taille actuelle du dictionnaire.
- Numéros réservés : **`<PAD>` = 0** et **`<UNK>` = 1**. Le premier vrai mot reçoit donc le 2.
- Le vocabulaire est construit **uniquement sur le train**, avec 8 289 mots. Sur le dev et le test, un mot inconnu devient `<UNK>` : le modèle n'a appris de vecteur que pour les mots du train.
- « L' » et « l' » ont deux numéros différents : pour le modèle, ce sont deux mots sans lien.

### Padding et troncage (`pad_tensor`)
- Un batch est une **matrice** : toutes les phrases doivent avoir la même longueur L = 40.
- Phrase trop courte : on la complète avec des 0. Phrase trop longue : on la coupe.
  `[4, 9]` avec L = 5 devient `[4, 9, 0, 0, 0]`.
- On padde **x et y**. C'est la différence avec la classification de texte du cours, où `y` est une seule étiquette par phrase.

---

## 4. Le modèle (`model.py`)

```python
self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
self.gru = nn.GRU(d_embed, d_hidden, batch_first=True, bias=False)
self.dropout = nn.Dropout(0.1)
self.decision = nn.Linear(d_hidden, d_out)
...
hidden = self.gru(embedded)[0]
```

| Option | Pourquoi |
|---|---|
| `padding_idx=0` | le vecteur du padding reste nul et n'est jamais appris |
| `batch_first=True` | nos tenseurs sont rangés (B, L), et non (L, B) |
| `bias=False` | sans biais, une entrée nulle (le padding) donne une sortie nulle |
| `Dropout(0.1)` | met 10 % des valeurs à zéro pendant l'entraînement, ce qui limite le surapprentissage |

### ⭐ La question clé : `gru(...)[0]` ou `[1]` ?
Le GRU renvoie deux résultats :
- `[0]` : l'état caché **à chaque mot**, de forme (B, L, d_hidden). On en a besoin pour **une prédiction par mot** : c'est l'étiquetage.
- `[1]` : l'état caché **après le dernier mot**, de forme (1, B, d_hidden), un résumé de la phrase. Il sert pour **une prédiction par phrase** : c'est la classification, dans le code du cours.

### Nombre de paramètres (≈ 890 000, fichier de 3,6 Mo)
| Couche | Taille | Paramètres |
|---|---|---|
| Embedding | 8 289 × 100 | 828 900 (93 % du modèle) |
| GRU | 2 × (300 × 100) | 60 000. Le 300 vient des 3 « portes » du GRU × 100 |
| Linear | 17 × 100 + 17 | 1 717 |

---

## 5. L'entraînement (`fit`)

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()                          # 1. remettre les gradients à 0
        logits = model(x)                              # 2. prédire
        loss = criterion(logits.transpose(1, 2), y)    # 3. mesurer l'erreur
        loss.backward()                                # 4. calculer les gradients
        optimizer.step()                               # 5. mettre à jour les paramètres
```

- **Epoch** : un passage complet sur tout le train.
- **Loss (cross-entropie)** : petite si le modèle donne un score élevé à la bonne étiquette.
- **`zero_grad`** : sinon, PyTorch **additionne** les gradients d'un batch à l'autre.
- **`backward`** : calcule le gradient de la loss pour chaque paramètre (rétro-propagation).
- **`step`** : Adam modifie chaque paramètre dans le sens qui fait baisser la loss.
- **`transpose(1, 2)`** : `CrossEntropyLoss` attend (B, nb_classes, L), alors que le modèle sort (B, L, nb_classes).
- **`ignore_index=PAD_ID`** : le padding ne compte pas dans la loss. Sinon, le modèle apprendrait surtout à prédire du padding.

### Évaluation sur le dev (`perf`)
```python
model.eval()                       # désactive le dropout
with torch.no_grad():              # pas besoin de gradients
    t_hat = logits.argmax(dim=2)   # étiquette prédite
    mask = (y != PAD_ID)           # on ignore le padding
    correct += ((t_hat == y) * mask).sum()
```
Accuracy = bonnes réponses / nombre de **vrais** mots.

### Garder le meilleur modèle
À chaque epoch, si la `dev_loss` est la plus basse vue jusque-là, on copie les paramètres avec `copy.deepcopy(model.state_dict())`. À la fin, on les recharge. Sans `deepcopy`, on garderait une simple référence qui continuerait de changer.

### Graine du hasard (`--seed 42`)
L'initialisation des poids et le mélange (`shuffle`) sont aléatoires. Sans graine, deux lancements identiques ont donné **90,9 %** puis **87,9 %** sur le dev. Avec `Util.init_seed(42)`, les résultats sont reproductibles.
→ Pour comparer deux réglages sérieusement, il faut plusieurs lancements, pas un seul.

---

## 6. La sauvegarde et la prédiction

### Que sauvegarde-t-on dans `postag.pt` ?
1. **Les paramètres** (`state_dict`) : ce que le modèle a appris.
2. **Les vocabulaires** : sans eux, impossible de convertir mot → numéro à l'entrée et numéro → étiquette à la sortie. Le modèle serait inutilisable.
3. **Les hyper-paramètres** : pour recréer un réseau vide de la bonne taille avant `load_state_dict`.

### Prédiction (`predict_postag.py`)
- Phrase par phrase, **sans padding ni troncage** : pas de batch, donc pas besoin d'aligner les longueurs. Et il faut une étiquette pour **chaque** mot. Le GRU accepte des séquences de n'importe quelle longueur.
- `torch.LongTensor([ids])` : un « batch » d'une seule phrase, de forme (1, n).
- `Util.rev_vocab` : retrouve l'étiquette à partir de son numéro (2 → NOUN).
- `torch.load(..., weights_only=False)` : nécessaire parce que le fichier contient aussi des dictionnaires, pas seulement des tenseurs.

---

## 7. Les résultats

### Courbe d'apprentissage (`resultats/train.log`, seed 42)
| Epoch | train_loss | dev_loss | dev_acc |
|---|---|---|---|
| 1 | 1,54 | 0,96 | 73,7 % |
| 5 | 0,38 | 0,45 | 87,0 % |
| **10** | **0,13** | **0,363 (min)** | **90,0 %** |
| 15 | 0,05 | 0,38 | 90,5 % |
| 20 | 0,02 | 0,42 | 90,2 % |

**À dire :**
- La loss du train baisse toujours : le modèle apprend.
- La loss du dev atteint son minimum à l'epoch 10, puis **remonte** : c'est le **surapprentissage**. Le modèle apprend le train par cœur.
- On garde donc le modèle de l'epoch 10.
- *Pourquoi la dev_loss monte alors que la dev_acc stagne ?* Le modèle devient trop sûr de lui, y compris quand il se trompe : la cross-entropie punit fort les erreurs commises avec confiance.

### Score final sur le test (`resultats/eval_test.txt`)
| | Accuracy |
|---|---|
| **Tous les mots** | **90,43 %** (8 679 / 9 598) |
| Mots inconnus (OOV) | 51,75 % (472 / 912) |
| Mots connus (calculé) | ≈ 94,5 % (8 207 / 8 686) |

→ Les mots inconnus ne représentent que 9,5 % des mots, mais causent **près de la moitié des erreurs** (440 sur 919).

*Pourquoi 90,0 % sur le dev pendant l'entraînement, alors que `evaluate.py` donne un chiffre un peu différent ?* Pendant l'entraînement, les phrases sont tronquées à 40 mots, donc les mots au-delà ne sont pas comptés. `evaluate.py` évalue les phrases entières.

### Exemples d'erreurs (dans `test.pred.conllu`)
- **« enregistrera » → PROPN** au lieu de VERB : c'est sûrement un mot inconnu. Le futur est rare dans le corpus, et le modèle ne voit qu'un vecteur `<UNK>`.
- **« une forte augmentation » : « forte » → NOUN** au lieu de ADJ : c'est le mot **suivant** qui lève l'ambiguïté, or le GRU ne lit que de gauche à droite.

---

## 8. Limites et améliorations possibles

| Limite | Pourquoi | Amélioration |
|---|---|---|
| Mots inconnus (52 %) | `<UNK>` n'apparaît jamais dans le train, donc son vecteur reste aléatoire | remplacer au hasard des mots rares du train par `<UNK>` pendant l'entraînement |
| Contexte limité | le GRU ne voit que les mots **précédents** | GRU **bidirectionnel** (`bidirectional=True`), qui voit aussi les mots suivants |
| Le mot lui-même est ignoré | un mot n'est qu'un numéro, sans orthographe | utiliser les **caractères** (majuscule → PROPN, « -ment » → ADV) |
| Majuscules | « L' » et « l' » sont deux mots différents | tout mettre en minuscules (mais on perd l'indice des noms propres) |

---

## 9. Questions flash

- **Pourquoi un RNN plutôt qu'un dictionnaire ?** Les mots sont ambigus (« a » est AUX ou VERB, « différents » est DET ou ADJ) : il faut le contexte.
- **À quoi sert l'embedding ?** À transformer un numéro, qui ne veut rien dire, en un vecteur appris. Des mots proches finissent avec des vecteurs proches.
- **À quoi sert `<PAD>` ? Et `<UNK>` ?** `<PAD>` complète les phrases courtes dans un batch. `<UNK>` remplace les mots absents du train.
- **Pourquoi des batches ?** Pour calculer sur plusieurs phrases en parallèle, ce qui est plus rapide et rend l'apprentissage plus stable.
- **Pourquoi `shuffle=True` sur le train, `False` sur le dev ?** Pour que le modèle ne voie pas toujours les phrases dans le même ordre. Sur le dev, on ne fait qu'évaluer, donc l'ordre n'a pas d'importance.
- **Qu'est-ce que le surapprentissage, et comment le voit-on ?** Le modèle apprend le train par cœur : la loss du train baisse pendant que celle du dev remonte.
- **Qu'est-ce qu'un logit ?** Un score brut, avant softmax. Le softmax est intégré dans `CrossEntropyLoss`.
- **Différence avec le classifieur du cours ?** `[0]` au lieu de `[1]`, `y` paddé, `transpose` avant la loss, et une accuracy calculée mot par mot avec un masque.

---

## 10. Commandes pour tout relancer

```bash
SEQ=../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple
python train_postag.py --train $SEQ.train --dev $SEQ.dev --model resultats/postag.pt
python predict_postag.py --model resultats/postag.pt --input $SEQ.test --output resultats/test.pred.conllu
python ../../pstal-etu/lib/evaluate.py -p resultats/test.pred.conllu -g $SEQ.test -t $SEQ.train
```
