import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
from torch.autograd import Variable
import os

classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def create_folders():
    if not os.path.exists('./images'):
        os.mkdir('./images')


def plot_dataset(args, loader, title):
    n_images = args.show_n_images
    images, labels = next(iter(loader))
    plt.figure(figsize=(12, 8), facecolor='w')
    for i in range(n_images):
        ax = plt.subplot(4, int(n_images / 3), i + 1)
        plt.imshow(images[i, 0, :, :], vmin=0, vmax=1, cmap='gray')
        ax.set_title("{}".format(classes[labels[i]]), fontsize=15)
        plt.axis('off')

    plt.savefig('./images/FashionMNIST_samples_{}.png'.format(title), bbox_inches='tight')
    plt.show()


def plot_train_val_test_loss(epochs, train_losses, valid_losses, test_epoch, test_loss):
    plt.figure(figsize=(8, 4))
    ax = range(1, epochs + 1)
    plt.plot(ax, train_losses, label="Training loss")
    plt.plot(ax, valid_losses, label="Validation loss")
    plt.plot(test_epoch, test_loss, 'r^', label="Test loss")
    plt.legend(frameon=False)
    plt.title("Training and validation losses")
    plt.savefig('./images/train_val_loss.png', bbox_inches='tight')
    plt.show()


def plot_val_test_accuracy(epochs, valid_accuracy, test_epoch, test_accuracy):
    ax = range(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ax, valid_accuracy, label="Validation accuracy")
    plt.plot(test_epoch, test_accuracy, 'r^', label="Test accuracy")
    plt.legend(frameon=False)
    plt.title("Validation accuracy")
    plt.savefig('./images/train_val_accuracy.png', bbox_inches='tight')
    plt.show()


def plot_acc_for_each_class(args, model, test_loader, device):
    # Accuracy per class
    class_correct = [0. for _ in range(len(classes))]
    total_correct = [0. for _ in range(len(classes))]

    with torch.no_grad():
        model.eval()
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            test = Variable(inputs)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == targets).squeeze()

            for j in range(args.batch_size):
                label = targets[j]
                class_correct[label] += c[j].item()
                total_correct[label] += 1

    classes_accuracy = []
    for i in range(len(classes)):
        classes_accuracy.append(class_correct[i] * 100 / total_correct[i])
        print("Accuracy of {}: {:.2f}%".format(classes[i], class_correct[i] * 100 / total_correct[i]))

    fig, ax = plt.subplots()
    bar_plot = plt.bar(classes, classes_accuracy, tick_label=classes,
                       color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    def add_bar_value():
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, classes_accuracy[idx], ha='center',
                    va='bottom', rotation=0)

    add_bar_value()
    plt.ylim(0, 115)
    plt.title("Accuracy for each class [%]")
    plt.savefig('./images/acc_per_class.png', bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(targets, predicted):
    # Transform to list of prediction and targets
    prediction_l = [predicted[i].tolist() for i in range(len(predicted))]
    targets_l = [targets[i].tolist() for i in range(len(targets))]
    prediction_l = list(chain.from_iterable(prediction_l))
    targets_l = list(chain.from_iterable(targets_l))

    cm = confusion_matrix(targets_l, prediction_l)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, ax=ax, fmt="d")
    # labels and ticks
    ax.set_xlabel('Predicted targets')
    ax.set_ylabel('True targets')
    ax.xaxis.set_ticklabels(classes, rotation=90)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    plt.savefig('./images/confusion_matrix.png', bbox_inches='tight')
    plt.show()
