import argparse
from FashionMNIST.solver import Solver, ModelCheckpoint
from FashionMNIST.net import SimpleFashionNet, DeepFashionNet, SwishFashionNet
from FashionMNIST.utils import *
import os
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--network", type=str, choices=['SimpleNet', 'DeepNet', 'SwishNet'], default='SimpleNet')
    parser.add_argument("--optimizer", type=str, choices=['Adam', 'SGD', 'Adagrad'], default='Adam')
    parser.add_argument("--show_n_images", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)  # using mini-batches of 100 samples
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    solver = Solver(args)
    model, device, valid_loader, test_loader = solver.__getitem__()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    model_checkpoint = ModelCheckpoint(args.log_dir + "/best_model.pt", model)

    # Training model
    start = time.time()
    train_losses, valid_losses, valid_accuracy = [], [], []
    for epoch in range(args.max_epochs):
        train_loss = solver.train()
        val_loss, val_accuracy, _, _ = solver.test(model, valid_loader)
        print("Epoch: {}/{} | ".format(epoch + 1, args.max_epochs),
              "Training loss: {:.4f} | ".format(train_loss),
              "Validation loss {:.4f} | ".format(val_loss),
              "Validation accuracy {:.4f} | ".format(val_accuracy))
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        valid_accuracy.append(val_accuracy)
        model_checkpoint.update(epoch + 1, val_loss)

    # Test tre trained model
    model_path = args.log_dir + "/best_model.pt"
    if args.network == 'SimpleNet':
        model = SimpleFashionNet()
    elif args.network == 'DeepNet':
        model = DeepFashionNet()
    elif args.network == 'SwishNet':
        model = SwishFashionNet()
    else:
        print("Invalid Network, exit..")
        exit(1)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    test_loss, test_accuracy, prediction_list, targets_list = solver.test(model, test_loader)
    print("Test loss:  {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))
    end = time.time()
    print("Trained model on {} epochs in {:.2f} minutes".format(args.max_epochs, (end - start) / 60))
    # Plot evaluations
    plot_train_val_test_loss(args.max_epochs, train_losses, valid_losses, model_checkpoint.epoch, test_loss)
    plot_val_test_accuracy(args.max_epochs, valid_accuracy, model_checkpoint.epoch, test_accuracy)
    plot_acc_for_each_class(args, model, test_loader, device)
    plot_confusion_matrix(prediction_list, targets_list)


if __name__ == "__main__":
    main()
