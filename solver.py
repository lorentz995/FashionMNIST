from FashionMNIST.dataset import Dataset
from FashionMNIST.net import FashionNet1
import torch
import torch.optim as optim


class Solver():
    def __init__(self, args):
        # loading data loaders
        self.train_loader, self.valid_loader, self.test_loader = Dataset(args).__getitem__()
        # Use CUDA if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\nDevice: {}\n".format(self.device))
        self.model = FashionNet1()
        self.model.to(self.device)
        print(self.model)

        # Instantiating Cross Entropy loss and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)  # Using Adam optimizer
