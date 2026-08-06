import numpy as np
from data import load_data
from model import BiLM

rng = np.random.default_rng(42)

sentences_char_ids, sentences_word_ids, char2idx, word2idx, vocab_words = load_data(
    sentence_len=20, min_char_len=3
)
char_vocab_size = len(char2idx) + 1  # +1 for PAD id 0
word_vocab_size = len(vocab_words)
print(f"Char vocab: {char_vocab_size} | Word vocab: {word_vocab_size} | Sentences: {len(sentences_char_ids)}")

CHAR_EMBED_DIM = 16
FILTER_WIDTH = 3
NUM_FILTERS = 32
HIDDEN_DIM = 32
EPOCHS = 5
LR = 0.1

model = BiLM(char_vocab_size, word_vocab_size, CHAR_EMBED_DIM, FILTER_WIDTH,
             NUM_FILTERS, HIDDEN_DIM, seed=0)

n_sentences = len(sentences_char_ids)

for epoch in range(EPOCHS):
    order = rng.permutation(n_sentences)
    total_loss = 0.0
    for idx in order:
        char_id_seqs = sentences_char_ids[idx]
        word_ids = sentences_word_ids[idx]
        cache = model.forward(char_id_seqs, word_ids)
        loss = model.loss(cache, word_ids)
        grads = model.backward(cache, word_ids)
        model.step(grads, LR)
        total_loss += loss
    avg_loss = total_loss / n_sentences
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.2f}")

np.savez("trained_bilm_lstm_fwd.npz", **model.fwd_lstm.params)
np.savez("trained_bilm_lstm_bwd.npz", **model.bwd_lstm.params)
np.savez("trained_bilm_highway.npz", **model.highway.params)
np.savez("trained_bilm_charcnn.npz", char_embeddings=model.charcnn.char_embeddings,
         W=model.charcnn.W, b=model.charcnn.b)
np.savez("trained_bilm_output.npz", U=model.U, b=model.b)
import pickle
with open("vocab_bilm.pkl", "wb") as f:
    pickle.dump({"char2idx": char2idx, "word2idx": word2idx, "vocab_words": vocab_words}, f)
print("Saved all trained_bilm_*.npz files and vocab_bilm.pkl")
