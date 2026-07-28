import numpy as np
import pickle

with open("vocab_w2v.pkl", "rb") as f:
    vocab = pickle.load(f)
word2idx, vocab_words = vocab["word2idx"], vocab["vocab_words"]

params = np.load("trained_w2v.npz")
C_in = params["C_in"]  # the "useful" table in practice


def nearest_neighbors(word, k=8):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    vec = C_in[word2idx[word]]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        norms = np.linalg.norm(C_in, axis=1) * np.linalg.norm(vec) + 1e-10
        sims = (C_in @ vec) / norms
    top = np.argsort(-sims)[1:k + 1]
    print(f"Nearest neighbors of '{word}':")
    for idx in top:
        print(f"  {vocab_words[idx]:<15s} cos_sim={sims[idx]:.3f}")


if __name__ == "__main__":
    for w in ["alice", "queen", "said", "little", "she", "king"]:
        print()
        nearest_neighbors(w)