import numpy as np
from data import tokenize, build_vocab, build_cbow_pairs, build_negative_sampler
from model import Word2Vec

rng = np.random.default_rng(42)

text = open("corpus_raw.txt").read()
tokens = tokenize(text)
word2idx, vocab_words, word_counts = build_vocab(tokens)
V = len(vocab_words)
input_ids, target_ids = build_cbow_pairs(tokens, word2idx, window_size=2)
neg_probs = build_negative_sampler(word_counts)
print(f"Vocab size: {V} | CBOW pairs: {len(input_ids)}")

EMBED_DIM, K, BATCH_SIZE, EPOCHS, LR = 32, 5, 128, 20, 0.5
model = Word2Vec(vocab_size=V, embed_dim=EMBED_DIM, mode="cbow", seed=0)

for epoch in range(EPOCHS):
    perm = rng.permutation(len(input_ids))
    inp, tgt = input_ids[perm], target_ids[perm]
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(inp), BATCH_SIZE):
        batch_in = inp[start:start + BATCH_SIZE]
        batch_tgt = tgt[start:start + BATCH_SIZE]
        batch_neg = rng.choice(V, size=(len(batch_in), K), p=neg_probs)
        p_pos, p_neg, cache = model.forward(batch_in, batch_tgt, batch_neg)
        loss = model.loss(p_pos, p_neg)
        grads = model.backward(cache, batch_in, batch_tgt, batch_neg)
        model.step(grads, LR)
        total_loss += loss
        n_batches += 1
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1}/{EPOCHS}: avg loss = {total_loss/n_batches:.4f}")

print("\nNearest neighbors (CBOW):")
C_in = model.C_in
for w in ["queen", "said", "alice"]:
    if w not in word2idx:
        continue
    vec = C_in[word2idx[w]]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        norms = np.linalg.norm(C_in, axis=1) * np.linalg.norm(vec) + 1e-10
        sims = (C_in @ vec) / norms
    top = np.argsort(-sims)[1:6]
    print(f"  {w}: {[vocab_words[i] for i in top]}")