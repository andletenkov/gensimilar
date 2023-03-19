from pathlib import Path

from data_utils import WordsDataset
from nn import MLP, Linear, Tanh

if __name__ == "__main__":
    context_length = 5
    embedding_size = 10
    n_hidden = 100

    dataset = WordsDataset(Path("surnames.txt"))
    mlp = MLP(
        [
            Linear(context_length * embedding_size, n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            Tanh(),
            Linear(n_hidden, n_hidden),
            Tanh(),
            Linear(n_hidden, dataset.vocab_size),
        ],
        data=dataset.train_val_test_split(0.8, context_length),
        vocab_size=dataset.vocab_size,
        emb_size=embedding_size,
    )
    mlp.fit()
    mlp.plot_activation_distribution()
