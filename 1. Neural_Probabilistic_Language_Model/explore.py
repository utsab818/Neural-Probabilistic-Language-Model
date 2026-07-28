import numpy as np
import pickle
from model import NPLM
from data import CONTEXT_SIZE, tokenize

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
word2idx, vocab_words = vocab["word2idx"], vocab["vocab_words"]
V = len(vocab_words)

params = np.load("trained_params.npz")
model = NPLM(vocab_size=V, context_size=CONTEXT_SIZE, embed_dim=params["C"].shape[1],
             hidden_dim=params["H"].shape[0], seed=0)
model.C, model.H, model.d, model.U, model.b = (
    params["C"], params["H"], params["d"], params["U"], params["b"]
)


def nearest_neighbors(word, k=8):
    """
    Find the k words whose embedding vectors (rows of model.C) are most
    similar to the given word's embedding, using cosine similarity:

        cos_sim(u, v) = (u . v) / (||u|| * ||v||)

    Steps:
      1. look up the word's row in model.C (handle "word not in word2idx"
         by printing a message and returning early)
      2. compute cosine similarity between that row and EVERY row of model.C
         (vectorize this -- don't loop over 1119 words one at a time)
      3. sort by similarity, descending, and take the top k -- but skip the
         word itself (it will always be the most "similar" to itself,
         with similarity 1.0, which isn't an interesting neighbor)
      4. print each neighbor word and its similarity score
    """
    if word not in word2idx:
        print(f"Word '{word}' not found in vocabulary.")
        return

    word_idx = word2idx[word]
    word_vector = model.C[word_idx]
    dot_products = model.C @ word_vector
    norms = np.linalg.norm(model.C, axis=1) * np.linalg.norm(word_vector)
    cosine_similarities = dot_products / (norms + 1e-10)
    top_k_indices = np.argsort(-cosine_similarities)
    neighbors = []
    for idx in top_k_indices:
        if idx != word_idx:
            neighbors.append((vocab_words[idx], cosine_similarities[idx]))
        if len(neighbors) == k:
            break

    for neighbor_word, similarity in neighbors:
        print(f"  {neighbor_word}: {similarity:.4f}")


def generate(seed_text, n_words=30, temperature=0.8, seed=0):
    """
    Autoregressively generate n_words words, starting from seed_text.

    Steps:
      1. tokenize(seed_text) to get starting tokens, map to ids via word2idx
         (missing words -> 0, the <unk> id)
      2. if there are fewer than CONTEXT_SIZE tokens so far, left-pad the id
         list with 0s (<unk>) so there's always a full context window to feed
         the model
      3. loop n_words times:
         a. take the last CONTEXT_SIZE ids as this step's context
         b. run model.forward() on it (shape (1, CONTEXT_SIZE), a batch of 1)
         c. apply temperature: divide the log-probabilities by `temperature`
            before re-normalizing (low temperature -> more confident/greedy,
            high temperature -> more random) -- then renormalize into a
            valid probability distribution again (subtract max before exp,
            same numerical-stability trick as in forward())
         d. sample the next id from this distribution (np.random.default_rng
            has a .choice(V, p=probs) method for this)
         e. append the sampled id to your running list of ids
      4. convert the final id list back to words and join into a string

    Return: the generated string.
    """
    rng = np.random.default_rng(seed)
    tokens = tokenize(seed_text)
    context_ids = [word2idx.get(t, 0) for t in tokens]
    context_ids = [0] * max(0, CONTEXT_SIZE - len(context_ids)) + context_ids[-CONTEXT_SIZE:]
    generated_ids = context_ids.copy()

    for _ in range(n_words):
        context_array = np.array(generated_ids[-CONTEXT_SIZE:]).reshape(1, -1)
        probs, _ = model.forward(context_array)
        log_probs = np.log(probs + 1e-10) / temperature
        exp_probs = np.exp(log_probs - np.max(log_probs))
        normalized_probs = exp_probs / np.sum(exp_probs)
        next_id = rng.choice(V, p=normalized_probs.ravel())
        generated_ids.append(next_id)

    generated_tokens = [vocab_words[idx] for idx in generated_ids]
    return " ".join(generated_tokens)


if __name__ == "__main__":
    print("=" * 60)
    print("NEAREST NEIGHBORS")
    print("=" * 60)
    for w in ["alice", "queen", "said", "little"]:
        print()
        nearest_neighbors(w)

    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)
    for seed_text in ["alice was", "the queen said"]:
        print(f"\nseed='{seed_text}'")
        print(" ", generate(seed_text, n_words=20))