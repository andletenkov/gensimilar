from pathlib import Path
import random

import torch


class WordsDataset:
    def __init__(self, data_path: Path) -> None:
        self.words = [
            line.strip() for line in data_path.open(encoding="utf-8").readlines()
        ]
        random.shuffle(self.words)

        self.tokens = sorted(list(set("".join(self.words) + ".")))
        self.token_map = {ch: i for i, ch in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

    def _words_to_tensors(
        self,
        words: list[str],
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = [self.token_map["."]] * context_length
        x, y = [], []

        for word in words:
            for c in word + ".":
                i = self.token_map[c]
                x.append(ctx)
                y.append(i)
                ctx = ctx[1:] + [i]

        return torch.tensor(x), torch.tensor(y)

    def train_val_test_split(
        self,
        frac: float = 0.8,
        context_length: int = 5,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        train_slice = int(frac * len(self.words))
        val_slice = int((frac + (1 - frac) / 2) * len(self.words))

        x_train, y_train = self._words_to_tensors(
            self.words[:train_slice], context_length
        )
        x_val, y_val = self._words_to_tensors(
            self.words[train_slice:val_slice], context_length
        )
        x_test, y_test = self._words_to_tensors(self.words[val_slice:], context_length)

        return x_train, y_train, x_val, y_val, x_test, y_test
