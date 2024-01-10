import torch

class TestModel(torch.nn.Module):
    """
    A mock neural network for testing purposes, not even trainable (simulates an already
    trained model)
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super(TestModel, self).__init__()

        layers = [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        for i in range(0, len(hidden_dims) - 1):
            layers += [torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden_dims[-1], output_dim)]

        self.network = torch.nn.Sequential(*layers)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)