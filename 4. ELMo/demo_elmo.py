import numpy as np
import pickle
from model import BiLM
from scalar_mix import ScalarMix
from data import tokenize, word_to_char_ids

with open("vocab_bilm.pkl", "rb") as f:
    v = pickle.load(f)
char2idx, word2idx, vocab_words = v["char2idx"], v["word2idx"], v["vocab_words"]

char_vocab_size = len(char2idx) + 1
word_vocab_size = len(vocab_words)

model = BiLM(char_vocab_size, word_vocab_size, char_embed_dim=16, filter_width=3,
             num_filters=32, hidden_dim=32, seed=0)

fwd = np.load("trained_bilm_lstm_fwd.npz")
for k in model.fwd_lstm.params:
    model.fwd_lstm.params[k] = fwd[k]
bwd = np.load("trained_bilm_lstm_bwd.npz")
for k in model.bwd_lstm.params:
    model.bwd_lstm.params[k] = bwd[k]
hw = np.load("trained_bilm_highway.npz")
for k in model.highway.params:
    model.highway.params[k] = hw[k]
cnn = np.load("trained_bilm_charcnn.npz")
model.charcnn.char_embeddings = cnn["char_embeddings"]
model.charcnn.W = cnn["W"]
model.charcnn.b = cnn["b"]
out = np.load("trained_bilm_output.npz")
model.U, model.b = out["U"], out["b"]


def get_elmo_layers(sentence_words, min_char_len=3):
    char_id_seqs = [word_to_char_ids(w, char2idx, min_char_len) for w in sentence_words]
    word_ids = np.array([word2idx.get(w, 0) for w in sentence_words])
    cache = model.forward(char_id_seqs, word_ids)
    return cache["X"], cache["H_fwd"], cache["H_bwd"]


if __name__ == "__main__":
    sent1 = tokenize("i sat by the river bank")
    sent2 = tokenize("i deposited money at the bank")

    X1, Hf1, Hb1 = get_elmo_layers(sent1)
    X2, Hf2, Hb2 = get_elmo_layers(sent2)

    idx1 = sent1.index("bank")
    idx2 = sent2.index("bank")

    # h_0 = charCNN+highway output (context-independent, purely spelling-based)
    h0_1, h0_2 = X1[idx1], X2[idx2]
    print("h_0 identical across both sentences (purely spelling-based):",
          np.allclose(h0_1, h0_2))

    # h_1 = concat(forward, backward) LSTM hidden state (genuinely contextual)
    h1_1 = np.concatenate([Hf1[idx1], Hb1[idx1]])
    h1_2 = np.concatenate([Hf2[idx2], Hb2[idx2]])
    print("h_1 identical across both sentences:", np.allclose(h1_1, h1_2))
    print("h_1 cosine similarity between the two 'bank' contexts:",
          np.dot(h1_1, h1_2) / (np.linalg.norm(h1_1) * np.linalg.norm(h1_2) + 1e-10))

    # combine with ScalarMix (h_0 padded to same dim as h_1 for this toy demo
    # by concatenating with itself -- a real setup would project dims to match)
    h0_1_padded = np.concatenate([h0_1, h0_1])
    h0_2_padded = np.concatenate([h0_2, h0_2])
    mix = ScalarMix(num_layers=2, seed=0)
    elmo_1, _ = mix.forward([h0_1_padded, h1_1])
    elmo_2, _ = mix.forward([h0_2_padded, h1_2])
    print("\nFinal ELMo_word representations differ across the two sentences:",
          not np.allclose(elmo_1, elmo_2))
