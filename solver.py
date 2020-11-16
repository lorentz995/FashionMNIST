from FashionMNIST.dataset import Dataset
from FashionMNIST.net import SimpleFashionNet, DeepFashionNet, SwishFashionNet
import torch
import torch.optim as optim


class Solver:
    def __init__(self, args):
        # loading data loaders
        self.train_loader, self.valid_loader, self.test_loader = Dataset(args).__getitem__()
        # Use CUDA if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\nDevice: {}\n".format(self.device))
        if args.network == 'SimpleNet':
            self.model = SimpleFashionNet()
        elif args.network == 'DeepNet':
            self.model = DeepFashionNet()
        elif args.network == 'SwishNet':
            self.model = SwishFashionNet()
        else:
            print("Invalid Network, exit..")
            exit(1)

        self.model.to(self.device)
        print(self.model)

        # Instantiating Cross Entropy loss and optimizer
        self.loss = None
        self.loss_function = torch.nn.CrossEntropyLoss()
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)  # Using Adam optimizer
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)  # SGD optimizer
        elif args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=args.learning_rate)  # Using Adagrad optimizer
        else:
            print("Invalid optimizer, exit..")
            exit(1)

    def __getitem__(self):
        return self.model, self.device, self.valid_loader, self.test_loader

    def train(self):
        self.model.train()
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Compute the forward pass through the network up to the loss
            outputs = self.model(inputs)
            self.loss = self.loss_function(outputs, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        return self.loss

    def test(self, model, loader):
        prediction_list = []
        targets_list = []
        with torch.no_grad():
            model.eval()
            N = 0
            tot_loss, correct = 0, 0
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_list.append(targets)
                outputs = model(inputs)

                # Accumulate the exact number of processed samples
                N += inputs.shape[0]
                tot_loss += inputs.shape[0] * self.loss_function(outputs, targets).item()

                predicted_targets = outputs.argmax(dim=1)
                prediction_list.append(predicted_targets)
                correct += (predicted_targets == targets).sum().item()
            return tot_loss / N, correct / N, prediction_list, targets_list


# Saving the best model that minimize the validation loss
class ModelCheckpoint:
    def __init__(self, filepath, model):
        self.min_loss = None
        self.epoch = None
        self.filepath = filepath
        self.model = model

    def update(self, epoch, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Found minimum validation loss: Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            self.min_loss = loss
            self.epoch = epoch
