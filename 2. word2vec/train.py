import numpy as np
from data import load_data
from model import Word2Vec

rng = np.random.default_rng(42)

center_ids, context_ids, neg_probs, word2idx, vocab_words = load_data()
V = len(vocab_words)
print(f"Vocab size: {V} | Skip-gram pairs: {len(center_ids)}")

EMBED_DIM = 32
NEGATIVE_SAMPLES = 5
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.5

model = Word2Vec(vocab_size=V, embed_dim=EMBED_DIM, seed=0)


def sample_negatives(batch_size, k):
    """
    Draw a (batch_size, k) array of negative sample word ids, according to
    neg_probs (the count^0.75 distribution), using rng.choice(V, size=..., p=...).
    """
    return rng.choice(V, size=(batch_size, k), p=neg_probs)


# For each epoch:
#   1. shuffle (center_ids, context_ids) together (same permutation for both --
#      they're paired rows, shuffling them differently would scramble which
#      center goes with which context!)
#   2. iterate in chunks of BATCH_SIZE
#   3. for each batch: sample negatives, forward -> loss -> backward -> step
#   4. track and print average loss per epoch
for epoch in range(EPOCHS):
    indices = np.arange(len(center_ids))
    rng.shuffle(indices)
    center_ids_shuffled = center_ids[indices]
    context_ids_shuffled = context_ids[indices]

    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(center_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_center = center_ids_shuffled[start:end]
        batch_context = context_ids_shuffled[start:end]
        batch_negatives = sample_negatives(len(batch_center), NEGATIVE_SAMPLES)

        p_pos, p_neg, cache = model.forward(batch_center, batch_context, batch_negatives)

        loss = model.loss(p_pos, p_neg)
        total_loss += loss
        num_batches += 1

        grads = model.backward(cache, batch_center, batch_context, batch_negatives)
        model.step(grads, LR)

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

np.savez("trained_w2v.npz", C_in=model.C_in, C_out=model.C_out)
import pickle
with open("vocab_w2v.pkl", "wb") as f:
    pickle.dump({"word2idx": word2idx, "vocab_words": vocab_words}, f)
print("Saved trained_w2v.npz and vocab_w2v.pkl")
