import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, args):
        toTensor = transforms.Compose([transforms.ToTensor()])
        self.train_valid_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir,
                                                                     train=True,
                                                                     transform=toTensor,
                                                                     download=True)

        self.train_valid_loader = torch.utils.data.DataLoader(dataset=self.train_valid_dataset,
                                                              batch_size=args.batch_size,
                                                              shuffle=True,
                                                              num_workers=4)

        # Showing some images of the dataset with the associated labels
        classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
        n_images = args.show_n_images
        images, labels = next(iter(self.train_valid_loader))
        plt.figure(figsize=(12, 8), facecolor='w')
        for i in range(n_images):
            ax = plt.subplot(4, int(n_images / 3), i + 1)
            plt.imshow(images[i, 0, :, :], vmin=0, vmax=1, cmap='gray')
            ax.set_title("{}".format(classes[labels[i]]), fontsize=15)
            plt.axis('off')

        # plt.savefig('FashionMNIST_samples.png', bbox_inches='tight')
        plt.show()

        if args.norm_data:  # if true, train_valid dataset will be normalized for its own mean and std
            print("\nNormalizing data for own mean and std\n")
            n_pixels = len(self.train_valid_dataset) * 28 * 28  # 28*28 is the height and width of the images inside
            # our dataset
            total_sum = 0
            # Calculating mean
            for batch in self.train_valid_loader: total_sum += batch[0].sum()
            mean = total_sum / n_pixels
            # Calculating std
            sum_of_squared_error = 0
            for batch in self.train_valid_loader:
                sum_of_squared_error += ((batch[0] - mean) ** 2).sum()
            std = torch.sqrt(sum_of_squared_error / n_pixels)
            print("Mean: {}, std: {}".format(mean, std))

            # Normalizing the datasets with mean and std just calculated
            self.normalized = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            self.train_valid_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir,
                                                                         train=True,
                                                                         transform=self.normalized,
                                                                         download=True)

        valid_ratio = 0.2
        # Split into training and validation sets
        self.train_dataset, self.valid_dataset = \
            torch.utils.data.dataset.random_split(self.train_valid_dataset,
                                                  [int((1.0 - valid_ratio) * len(self.train_valid_dataset)),
                                                   int(valid_ratio * len(self.train_valid_dataset))])

        # Download the test dataset normalized
        self.test_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir,
                                                              train=False,
                                                              transform=self.normalized if args.norm_data else toTensor,
                                                              download=True)
        # Load the train, valid and test loaders normalized
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

    def __getitem__(self):
        print("The training set contains {} images in {} batches"
              .format(len(self.train_loader.dataset), len(self.train_loader)))
        print("The validation set contains {} images in {} batches"
              .format(len(self.valid_loader.dataset), len(self.valid_loader)))
        print("The test set contains {} images in {} batches"
              .format(len(self.test_loader.dataset), len(self.test_loader)))

        return self.train_loader, self.valid_loader, self.test_loader
