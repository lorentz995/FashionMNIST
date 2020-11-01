import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torch
from torch.autograd import Variable

classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def plot_train_val_test_loss(epochs, train_losses, valid_losses, test_epoch, test_loss):
    plt.figure(figsize=(8, 4))
    ax = range(1, epochs + 1)
    plt.plot(ax, train_losses, label="Training loss")
    plt.plot(ax, valid_losses, label="Validation loss")
    plt.plot(test_epoch, test_loss, 'r^', label="Test loss")
    plt.legend(frameon=False)
    plt.title("Training and validation losses")
    plt.show()


def plot_val_test_accuracy(epochs, valid_accuracy, test_epoch, test_accuracy):
    ax = range(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ax, valid_accuracy, label="Validation accuracy")
    plt.plot(test_epoch, test_accuracy, 'r^', label="Test accuracy")
    plt.legend(frameon=False)
    plt.title("Validation accuracy")
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

    fig4 = plt.figure()
    plt.bar(classes, classes_accuracy,
            color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    plt.xticks(np.arange(len(classes)), classes)
    # for i, v in enumerate(classes_accuracy):
    #    ax.text(v+3, i +.25, str(v), color='blue', fontweigth='bold')
    plt.title("Accuracy for each class")
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
    plt.show()
