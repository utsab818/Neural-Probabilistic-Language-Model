import numpy as np
from data import load_data, CONTEXT_SIZE
from model import NPLM

rng = np.random.default_rng(42)

examples, word2idx, vocab_words, tokens = load_data(context_size=CONTEXT_SIZE)
V = len(vocab_words)

# held-out split
n = len(examples)
perm = rng.permutation(n)
examples = examples[perm]
n_val = max(1, int(0.05 * n))
val_examples, train_examples = examples[:n_val], examples[n_val:]

print(f"Vocab size: {V} | Train: {len(train_examples)} | Val: {len(val_examples)}")

EMBED_DIM = 16
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.08

model = NPLM(vocab_size=V, context_size=CONTEXT_SIZE, embed_dim=EMBED_DIM,
             hidden_dim=HIDDEN_DIM, seed=0)


def evaluate(data):
    """Compute (loss, perplexity) on a set of examples, no parameter updates."""
    context_ids = data[:, :-1]
    target_ids = data[:, -1]
    probs, _ = model.forward(context_ids)
    loss = model.loss(probs, target_ids)
    perplexity = np.exp(loss)
    return loss, perplexity


# TODO: main training loop
# For each epoch:
#   1. shuffle train_examples
#   2. iterate over it in chunks of BATCH_SIZE
#   3. for each batch: forward -> loss -> backward -> step
#   4. accumulate/print average training loss for the epoch
#   5. call evaluate(val_examples) and print val loss + perplexity
#
# Recall: perplexity = exp(loss) -- this connects directly to the NLL loss
# you derived; it's just "the loss re-expressed as an effective branching
# factor" (roughly: how many equally-likely words the model is choosing
# between, on average, when it's this uncertain).
for epoch in range(EPOCHS):
    shuffle = rng.permutation(len(train_examples))
    train_loss = 0
    for i in range(0, len(train_examples), BATCH_SIZE):
        batch = train_examples[shuffle[i:i + BATCH_SIZE]]
        context_ids = batch[:, :-1]
        target_ids = batch[:, -1]
        probs, cache = model.forward(context_ids)
        loss = model.loss(probs, target_ids)
        train_loss += loss * len(batch)
        grad = model.backward(cache, target_ids)
        model.step(grad, lr=LR)

    train_loss /= len(train_examples)
    val_loss, val_perplexity = evaluate(val_examples)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f}")

np.savez("trained_params.npz", C=model.C, H=model.H, d=model.d, U=model.U, b=model.b)
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump({"word2idx": word2idx, "vocab_words": vocab_words}, f)
print("Saved trained_params.npz and vocab.pkl")