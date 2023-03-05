import abc

import torch


class Layer(abc.ABC):
    def __init__(self):
        self._out = None

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
        self.training = True

        self.gamma = torch.ones(size)
        self.beta = torch.zeros(size)

        self.track_mean = torch.zeros(size)
        self.track_var = torch.ones(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_mean = x.mean(0, keepdim=True)
            x_var = x.var(0, keepdim=True)

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
