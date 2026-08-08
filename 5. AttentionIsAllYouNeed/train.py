import numpy as np
from data import make_dataset, PAD, BOS, EOS
from model import Transformer

rng = np.random.default_rng(42)

VOCAB_SIZE = 8
SEQ_LEN = 3
D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
NUM_LAYERS = 2
EPOCHS = 150
LR = 0.1
N_TRAIN = 100

dataset = make_dataset(N_TRAIN, SEQ_LEN, VOCAB_SIZE, seed=1)

model = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                     max_len=SEQ_LEN + 5, seed=0)

for epoch in range(EPOCHS):
    order = rng.permutation(len(dataset))
    total_loss = 0.0
    for idx in order:
        src, tgt_input, tgt_output = dataset[idx]
        probs, cache = model.forward(src, tgt_input)
        loss = model.loss(probs, tgt_output)
        grads = model.backward(cache, tgt_output)
        model.step(grads, LR)
        total_loss += loss
    avg_loss = total_loss / len(dataset)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")


def generate(model, src, max_len=SEQ_LEN + 2):
    """Greedy decoding: no teacher forcing, feed the model's own predictions
    back in, one token at a time."""
    encoder_out, _ = model.encode(src)
    tgt_ids = [BOS]
    for _ in range(max_len):
        y, _ = model.decode(np.array(tgt_ids), encoder_out)
        logits = y[-1] @ model.W_out + model.b_out
        next_id = int(np.argmax(logits))
        tgt_ids.append(next_id)
        if next_id == EOS:
            break
    return tgt_ids[1:]


print("\nGreedy decoding examples:")
test_examples = make_dataset(5, SEQ_LEN, VOCAB_SIZE, seed=99)
for src, _, _ in test_examples:
    pred = generate(model, src)
    expected = list(src[::-1]) + [EOS]
    print(f"  src={list(src)} -> pred={pred} | expected={expected}")
