import logging
import random
from pathlib import Path
from typing import Optional, Literal
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.DEBUG)
file = Path("surnames.txt")


class MLP:
    def __init__(
        self,
        data_path: Path,
        context_length: int = 3,
        embedding_size: int = 2,
        hidden_layer_size: int = 100,
        learning_rate: float = 0.12,
        num_steps: int = 1000,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        random_state: Optional[int] = 42,
    ):
        assert train_ratio < 1
        torch.random.manual_seed(random_state)
        random.seed(random_state)

        words = [line.strip() for line in data_path.open(encoding="utf-8").readlines()]
        random.shuffle(words)
        tokens = sorted(list(set("".join(words) + ".")))

        self._context_length = context_length
        self._token_map = {ch: i for i, ch in enumerate(tokens)}
        self._token_map_reversed = {v: k for k, v in self._token_map.items()}
        self._end_token = self._token_map["."]
        self._initial_context = [self._end_token] * self._context_length

        train_slice = int(train_ratio * len(words))
        val_slice = int((train_ratio + (1 - train_ratio) / 2) * len(words))
        self.x_train, self.y_train = self._prepare_dataset(words[:train_slice])
        self.x_val, self.y_val = self._prepare_dataset(words[train_slice:val_slice])
        self.x_test, self.y_test = self._prepare_dataset(words[val_slice:])

        self.emb = torch.randn(len(tokens), embedding_size)
        self.w1 = (
            torch.randn(context_length * embedding_size, hidden_layer_size)
            * (5 / 3)
            / context_length
            * embedding_size**0.5
        )  # kaiming init
        self.w2 = torch.randn(hidden_layer_size, self.emb.shape[0]) * 0.01
        self.b2 = torch.randn(self.emb.shape[0]) * 0

        self.bn_gain = torch.ones(1, hidden_layer_size)
        self.bn_bias = torch.zeros(1, hidden_layer_size)
        self.bn_mean = torch.zeros(1, hidden_layer_size)
        self.bn_std = torch.ones(1, hidden_layer_size)

        self.params = [self.emb, self.w1, self.w2, self.b2, self.bn_gain, self.bn_bias]

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, mode: Literal["train", "test", "val"] = "train") -> torch.Tensor:
        grad_state = torch.is_grad_enabled()
        x, y = {
            "train": (self.x_train, self.y_train),
            "val": (self.x_val, self.y_val),
            "test": (self.x_test, self.y_test),
        }[mode]

        if mode == "train":
            # use batches
            batch = torch.randint(0, self.x_train.shape[0], (self.batch_size,))
            x = x[batch]
            y = y[batch]

            xemb = self.emb[x]
            h_preact = xemb.view(-1, self.w1.shape[0]) @ self.w1
            mean = h_preact.mean(0)
            std = h_preact.std(0)

            # keep track of batch normalization mean/std
            with torch.no_grad():
                self.bn_mean = 0.999 * self.bn_mean + 0.001 * mean
                self.bn_std = 0.999 * self.bn_std + 0.001 * std
        else:
            torch.set_grad_enabled(False)
            xemb = self.emb[x]
            h_preact = xemb.view(-1, self.w1.shape[0]) @ self.w1
            mean = self.bn_mean
            std = self.bn_std

        try:
            h_preact = self.bn_gain * (h_preact - mean) / std + self.bn_bias
            h = torch.tanh(h_preact)
            logits = h @ self.w2 + self.b2

            # counts = logits.exp()
            # probs = counts / counts.sum(1, keepdim=True)
            # loss = -probs[torch.arange(self.y.shape[0]), self.y].log().mean()

            loss = F.cross_entropy(logits, y)
        finally:
            torch.set_grad_enabled(grad_state)
        return loss

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def fit(self):
        for p in self.params:
            p.requires_grad = True

        for i in range(self.num_steps):
            self.zero_grad()
            loss = self.forward("train")
            loss.backward()

            lr = round(
                self.learning_rate if i < 10000 else self.learning_rate / i * 5000, 3
            )
            if i % 10000 == 0:
                logging.info(
                    f"{i} / {self.num_steps} - Training loss: {loss:.4f}, lr: {lr}"
                )

            for p in self.params:
                p.data += -lr * p.grad

    def _prepare_dataset(self, words) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._initial_context
        x, y = [], []

        for word in words:
            for ch in word + ".":
                i = self._token_map[ch]
                x.append(ctx)
                y.append(i)
                ctx = ctx[1:] + [i]

        return torch.tensor(x), torch.tensor(y)

    def generate(self, count: int = 20):
        results = []

        for _ in range(count):
            out = []
            ctx = self._initial_context
            current = None

            while current != self._end_token:
                ctx_emb = self.emb[torch.tensor([ctx])]
                h_preact = ctx_emb.view(1, -1) @ self.w1
                h_preact = (
                    self.bn_gain * (h_preact - self.bn_mean) / self.bn_std
                    + self.bn_bias
                )
                h = torch.tanh(h_preact)
                logits = h @ self.w2 + self.b2
                probs = torch.softmax(logits, dim=1)

                current = torch.multinomial(probs, num_samples=1).item()
                ctx = ctx[1:] + [current]
                out.append(current)

            results.append("".join(self._token_map_reversed[i] for i in out[:-1]))

        return results

    def plot_embeddings(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.emb[:, 0].data, self.emb[:, 1].data, s=200)
        for i in range(self.emb.shape[0]):
            plt.text(
                self.emb[i, 0].item(),
                self.emb[i, 1].item(),
                self._token_map_reversed[i],
                ha="center",
                va="center",
                color="white",
            )
        plt.grid("minor")
        plt.show()


if __name__ == "__main__":
    mlp = MLP(
        file,
        num_steps=200000,
        embedding_size=10,
        hidden_layer_size=200,
        context_length=5,
    )
    mlp.fit()

    print(mlp.forward("test").item())
    print(mlp.forward("val").item())

    print(mlp.generate(100))
    # mlp.plot_embeddings()
