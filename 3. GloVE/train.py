import numpy as np
from data import load_data
from model import GloVe

rng = np.random.default_rng(42)

cooc, word2idx, vocab_words = load_data()
V = len(vocab_words)
print(f"Vocab size: {V} | Nonzero pairs: {len(cooc)}")

# Unpack the dict into flat arrays once -- much faster than iterating the
# dict every epoch.
pairs = list(cooc.items())
i_ids_all = np.array([p[0][0] for p in pairs], dtype=np.int32)
j_ids_all = np.array([p[0][1] for p in pairs], dtype=np.int32)
X_all = np.array([p[1] for p in pairs], dtype=np.float64)

EMBED_DIM = 32
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.05

model = GloVe(vocab_size=V, embed_dim=EMBED_DIM, seed=0)

# For each epoch:
#   1. shuffle i_ids_all, j_ids_all, X_all TOGETHER (same permutation for
#      all three -- they're aligned rows, like center_ids/context_ids were
#      for word2vec)
#   2. iterate in chunks of BATCH_SIZE
#   3. for each batch: forward -> loss -> backward -> step
#   4. track and print average loss per epoch
for epoch in range(EPOCHS): 
    shuffle_indices = rng.permutation(len(i_ids_all))
    i_ids_all = i_ids_all[shuffle_indices]
    j_ids_all = j_ids_all[shuffle_indices]
    X_all = X_all[shuffle_indices]
    total_loss = 0.0

    for start in range(0, len(i_ids_all), BATCH_SIZE):
        end = start + BATCH_SIZE
        i_ids = i_ids_all[start:end]
        j_ids = j_ids_all[start:end]
        X_ij = X_all[start:end]

        diff, f, cache = model.forward(i_ids, j_ids, X_ij)
        loss = model.loss(diff, f)
        grads = model.backward(cache)
        model.step(grads, LR)
        total_loss += loss * len(i_ids)

    avg_loss = total_loss / len(i_ids_all)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")