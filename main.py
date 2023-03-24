from pathlib import Path

import torch

from data_utils import WordsDataset
from nn import MLP, Linear, Tanh, BatchNorm1D


def generate(
    model: MLP, context_length: int, token_map: dict[str, int], count: int = 20
) -> list[str]:
    results = []
    index_to_char = {c: i for i, c in token_map.items()}
    end_token = token_map["."]

    model.eval()

    for _ in range(count):
        out = []
        ctx = [end_token] * context_length
        current = None

        while current != end_token:
            ctx_emb = model.emb[torch.tensor([ctx])]
            x = ctx_emb.view(1, -1)

            for layer in model.layers:
                x = layer(x)

            probs = torch.softmax(x, dim=1)

            current = torch.multinomial(probs, num_samples=1).item()
            ctx = ctx[1:] + [current]
            out.append(current)

        results.append("".join(index_to_char[i] for i in out[:-1]))

    return results


if __name__ == "__main__":
    context_length = 5
    embedding_size = 10
    n_hidden = 100

    dataset = WordsDataset(Path("surnames.txt"))
    mlp = MLP(
        [
            Linear(context_length * embedding_size, n_hidden),
            BatchNorm1D(n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            BatchNorm1D(n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            BatchNorm1D(n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            BatchNorm1D(n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            BatchNorm1D(n_hidden),
            Tanh(),
            Linear(n_hidden, dataset.vocab_size),
        ],
        data=dataset.train_val_test_split(0.8, context_length),
        vocab_size=dataset.vocab_size,
        emb_size=embedding_size,
    )
    mlp.fit()
    # mlp.plot_activation_distribution()

    new = generate(mlp, context_length, dataset.token_map)
    print(new)
