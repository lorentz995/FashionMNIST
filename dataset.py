import torch
import torchvision
import torchvision.transforms as transforms
from FashionMNIST.utils import create_folders, plot_dataset


class Transformer(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.dataset)


def compute_mean_std(dataset, loader):
    print("\nNormalizing training data for own mean and std")
    n_pixels = len(dataset) * 28 * 28  # 28*28 is the height and width of the images inside
    # our dataset
    total_sum = 0
    # Calculating mean
    for batch in loader:
        total_sum += batch[0].sum()
    mean = total_sum / n_pixels
    # Calculating std
    sum_of_squared_error = 0
    for batch in loader:
        sum_of_squared_error += ((batch[0] - mean) ** 2).sum()
    std = torch.sqrt(sum_of_squared_error / n_pixels)
    print("Mean: {}, std: {}\n".format(mean, std))
    return mean, std


class Dataset:
    def __init__(self, args):
        self.train_valid_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir,
                                                                     train=True,
                                                                     transform=transforms.ToTensor(),
                                                                     download=True)

        self.test_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir,
                                                              train=False,
                                                              transform=transforms.ToTensor(),
                                                              download=True)

        valid_ratio = 0.2
        # Split into training and validation sets
        self.train_dataset, self.valid_dataset = \
            torch.utils.data.dataset.random_split(self.train_valid_dataset,
                                                  [int((1.0 - valid_ratio) * len(self.train_valid_dataset)),
                                                   int(valid_ratio * len(self.train_valid_dataset))])

        # Showing some images of the dataset with the associated labels
        create_folders()
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,  # reshuffles the data at every epoch
                                                        num_workers=4,  # use 4 threads
                                                        drop_last=True)
        plot_dataset(args, self.train_loader, "Before_Normalization")

        mean, std = compute_mean_std(self.train_dataset, self.train_loader)
        # Normalizing the datasets with mean and std just calculated
        self.transform = transforms.Normalize(mean, std)

        # Transforms train and valid dataset into their normalized version
        self.train_dataset = Transformer(self.train_dataset, self.transform)
        self.valid_dataset = Transformer(self.valid_dataset, self.transform)
        self.test_dataset = Transformer(self.test_dataset, self.transform)

        # Create the data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,  # reshuffles the data at every epoch
                                                        num_workers=4,  # use 4 threads
                                                        drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(dataset=self.valid_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       drop_last=True)

        # Showing some images of the normalized dataset with the associated labels
        plot_dataset(args, self.train_loader, "Post_Normalization")

    def __getitem__(self):
        print("The training set contains {} images in {} batches"
              .format(len(self.train_loader.dataset), len(self.train_loader)))
        print("The validation set contains {} images in {} batches"
              .format(len(self.valid_loader.dataset), len(self.valid_loader)))
        print("The test set contains {} images in {} batches"
              .format(len(self.test_loader.dataset), len(self.test_loader)))

        return self.train_loader, self.valid_loader, self.test_loader
