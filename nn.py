import abc
import logging
from typing import Optional, Literal

import torch
import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt


class Layer(abc.ABC):
    def __init__(self):
        self._out = None
        self._training = True

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        raise NotImplementedError

    @property
    def out(self) -> torch.Tensor:
        return self._out

    @out.setter
    def out(self, value: torch.Tensor) -> None:
        self._out = value

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value


class Linear(Layer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def __init__(self, size_in: int, size_out: int, bias: bool = True):
        super().__init__()
        self.weights = torch.randn(size_in, size_out) / size_in**0.5
        self.bias = torch.randn(size_out) if bias else None

    def parameters(self) -> list[torch.Tensor]:
        if self.bias is not None:
            return [self.weights, self.bias]
        return [self.weights]


class BatchNorm1D(Layer):
    def __init__(self, size: int, eps: float = 1e-5, momentum: float = 0.01):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(size)
        self.beta = torch.zeros(size)

        self.track_mean = torch.zeros(size)
        self.track_var = torch.ones(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in (2, 3)
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            x_mean = x.mean(dim, keepdim=True)
            x_var = x.var(dim, keepdim=True)

            # update mean & variance buffers
            with torch.no_grad():
                self.track_mean = (
                    1 - self.momentum
                ) * self.track_mean + self.momentum * x_mean
                self.track_var = (
                    1 - self.momentum
                ) * self.track_var + self.momentum * x_var
        else:
            x_mean = self.track_mean
            x_var = self.track_var

        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_norm + self.beta
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh(Layer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []


class Embedding(Layer):
    def __init__(self, num_emb: int, emb_dim: int):
        super().__init__()
        self._weights = torch.randn((num_emb, emb_dim))

    def __call__(self, x: int | tuple[int, ...] | torch.Tensor) -> torch.Tensor:
        self.out = self._weights[x]
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self._weights]


class ConsecutiveFlatten(Layer):
    def __init__(self, n: int):
        super().__init__()
        self._n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x = x.view(b, t // self._n, c * self._n)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []


class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Linear):
                    # make the last layer less confident
                    if i == len(self.layers) - 1:
                        layer.weights *= 0.1
                    else:
                        # apply gain
                        layer.weights *= 5 / 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        for layer in self.layers:
            layer.training = value
        self._training = value

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]


class MLP:
    def __init__(
        self,
        model: Sequential,
        data: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ):
        self.model = model
        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        ) = data

    def parameters(self) -> list[torch.Tensor]:
        return self.model.parameters()

    def require_grad(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def eval(self):
        self.model.training = False

    def forward(
        self,
        mode: Literal["train", "test", "val"] = "train",
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        grad_state = torch.is_grad_enabled()

        x, y = {
            "train": (self.x_train, self.y_train),
            "val": (self.x_val, self.y_val),
            "test": (self.x_test, self.y_test),
        }[mode]

        if mode != "train":
            torch.set_grad_enabled(False)

        try:
            if batch_size is not None:
                batch = torch.randint(0, x.shape[0], (batch_size,))
                x = x[batch]
                y = y[batch]

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        finally:
            torch.set_grad_enabled(grad_state)

        return loss

    def fit(self, steps: int = 20000, lr: float = 0.1, batch_size: int = 32) -> None:
        self.model.training = True
        self.require_grad()

        for i in tqdm.tqdm(range(steps), desc="Train"):
            loss = self.forward("train", batch_size)

            self.zero_grad()
            loss.backward()

            lr = round(lr if i < 10000 else lr / 2, 3)
            if i % 1000 == 0:
                print(
                    f"{i:7d} / {steps:7d} - Training loss: {loss.item():.4f}, lr: {lr}"
                )

            for p in self.parameters():
                p.data += -lr * p.grad

        for layer in self.model.layers:
            print(layer.__class__.__name__, tuple(layer.out.shape))

    def plot_activation_distribution(self) -> None:
        plt.figure(figsize=(20, 4))
        legends = []
        for i, layer in enumerate(self.model.layers[:-1]):
            if not isinstance(layer, Tanh):
                continue
            t = layer.out
            legend = f"Layer #{i} ({layer.__class__.__name__})"
            print(
                f"{legend}: mean {t.mean():.2f}, std {t.std():.2f}, saturated {(t.abs() > 0.97).float().mean() * 100}%"
            )
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(legend)
        plt.legend(legends)
        plt.title("Activation distribution")
        plt.show()
