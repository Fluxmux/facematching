import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
import time
import sys
import numpy as np
from tqdm import tqdm


# user defined imports
from model import *
from helper_functions import *
from config import Config
from dataset_helper import *
from loss_function import *


def load_dataset():
    # loading dataset
    print("loading dataset...")
    if Config.MNIST:
        folder_dataset_training = dset.MNIST(root="./data", train=True, download=True)
        siamese_dataset_training = MnistDataset(
            imageFolderDataset=folder_dataset_training,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    elif Config.LFW:
        folder_dataset_training = dset.ImageFolder(root="./data/lfw/faces/training/")
        siamese_dataset_training = LFWDataset(
            imageFolderDataset=folder_dataset_training,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    else:
        folder_dataset_training = dset.ImageFolder(root=Config.training_dir)
        siamese_dataset_training = SiameseNetworkDataset(
            imageFolderDataset=folder_dataset_training,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    train_dataloader = DataLoader(
        siamese_dataset_training, shuffle=True, batch_size=Config.train_batch_size
    )

    if Config.MNIST:
        folder_dataset_validation = dset.MNIST(
            root="./data", train=False, download=True
        )
        siamese_dataset_validation = MnistDataset(
            imageFolderDataset=folder_dataset_validation,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    elif Config.LFW:
        folder_dataset_validation = dset.ImageFolder(root="./data/lfw/faces/testing/")
        siamese_dataset_validation = LFWDataset(
            imageFolderDataset=folder_dataset_validation,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    else:
        folder_dataset_validation = dset.ImageFolder(root=Config.validating_dir)
        siamese_dataset_validation = SiameseNetworkDataset(
            imageFolderDataset=folder_dataset_validation,
            transform=transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomRotation(Config.max_rotation),
                    transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                    transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                    transforms.RandomResizedCrop(100, scale=(0.85, 1), ratio=(1, 1)),
                    transforms.ToTensor(),
                ]
            ),
            should_invert=False,
        )

    valid_dataloader = DataLoader(
        siamese_dataset_validation, batch_size=Config.valid_batch_size, shuffle=True
    )
    print("dataset loaded!")
    Config.training_size = len(folder_dataset_training)
    print("size of training data:", len(folder_dataset_training), "samples")
    print("size of validation data:", len(folder_dataset_validation), "samples")

    return train_dataloader, valid_dataloader


def train(train_dataloader, valid_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Utilizing", device)
    # training
    if Config.model_type == "threshold":
        net = SiameseNetwork().to(device)
    elif Config.model_type == "absolute":
        net = SiameseNetworkAbs().to(device)
    elif Config.model_type == "resnet":
        net = SiameseResNet().to(device)
    elif Config.model_type == "resnet_cat":
        net = SiameseResNetConcat().to(device)
    elif Config.model_type == "resume_training":
        net = loadSiameseNetwork().to(device)
    else:
        sys.exit("'Config.model_type' wrongly specified!")

    if Config.loss_type == "contrastive":
        criterion = ContrastiveLoss()
    elif Config.loss_type == "cross_entropy":
        criterion = CrossEntropyLoss()

    print(net)
    print("Number of learnable parameters:", count_parameters(net))

    if Config.optimizer == "sgd":
        optimizer = optim.SGD(
            net.parameters(),
            lr=Config.learning_rate,
            momentum=0.99,
            dampening=0,
            weight_decay=Config.weight_decay,
            nesterov=True,
        )
    elif Config.optimizer == "adam":
        optimizer = optim.Adam(
            net.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
        )

    counter = []
    iterations = 5
    training_loss_history = []
    validation_loss_history = []
    iteration_number = 0
    valid_dataiter = iter(valid_dataloader)

    plt.figure(0)
    plt.title("Learning curve")
    plt.plot(counter, training_loss_history, color="blue", label="training")
    plt.plot(counter, validation_loss_history, color="orange", label="validation")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.grid(True, which="minor")
    start = time.time()

    if Config.loss_type == "contrastive":
        print("training with contrastive loss")
        for epoch in range(0, Config.train_number_epochs):
            progress = 0
            for i, train_data in enumerate(train_dataloader, 0):
                print(
                    "[ progress:", str(progress) + "/" + str(Config.training_size), "]"
                )
                net.train()
                # training data
                img0, img1, label = train_data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

                progress += img0.size(dim=0)

                optimizer.zero_grad()
                output1, output2 = net(img0, img1)

                loss = criterion(output1, output2, label)

                if i % iterations == 0:

                    net.eval()
                    with torch.no_grad():
                        # validation data
                        try:
                            valid_data = next(valid_dataiter)
                        except StopIteration:
                            valid_dataiter = iter(valid_dataloader)
                            valid_data = next(valid_dataiter)

                        img0, img1, label = valid_data
                        img0, img1, label = (
                            img0.to(device),
                            img1.to(device),
                            label.to(device),
                        )  # move the tensors to device
                        output1, output2 = net(img0, img1)
                        valid_loss = criterion(output1, output2, label)

                        print(
                            "Epoch number {}\nCurrent loss {}\nValid data loss {}\n".format(
                                epoch, loss.item(), valid_loss.item()
                            )
                        )
                        iteration_number += iterations
                        counter.append(iteration_number)

                        training_loss_history.append(loss.item())
                        validation_loss_history.append(valid_loss.item())

                        plt.plot(counter, training_loss_history, color="blue")
                        plt.plot(counter, validation_loss_history, color="orange")
                        plt.grid()
                        plt.savefig("lcurve_dynamic.svg", format="svg", dpi=1200)

                loss.backward()
                optimizer.step()

            # dynamic learning rate
            if Config.dynamic_lr != 0 and epoch % Config.dynamic_lr == 0 and epoch != 0:
                Config.learning_rate /= 2
                print("[learning rate adjusted to:", Config.learning_rate, "]")
                # also save the model
            if epoch % Config.save_rate == 0 and epoch != 0:
                saveSiameseNetwork(net, epoch)
                print("[model saved]")

    elif Config.loss_type == "cross_entropy":
        print("training with cross entropy loss")
        for epoch in range(0, Config.train_number_epochs):
            for i, train_data in enumerate(train_dataloader, 0):
                # training data
                net.train()
                img0, img1, label = train_data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

                optimizer.zero_grad()
                prediction = net(img0, img1)

                label.squeeze(1)
                loss = criterion(prediction, label)

                if i % iterations == 0:
                    net.eval()
                    with torch.no_grad():
                        # validation data
                        try:
                            valid_data = next(valid_dataiter)
                        except StopIteration:
                            valid_dataiter = iter(valid_dataloader)
                            valid_data = next(valid_dataiter)
                        img0, img1, label = valid_data
                        img0, img1, label = (
                            img0.to(device),
                            img1.to(device),
                            label.to(device),
                        )  # move the tensors to the device
                        prediction = net(img0, img1)
                        valid_loss = criterion(prediction.float(), label.float())

                        print(
                            "Epoch number {}\nCurrent loss {}\nValid data loss {}\n".format(
                                epoch, loss.item(), valid_loss.item()
                            )
                        )
                        iteration_number += iterations
                        counter.append(iteration_number)

                        training_loss_history.append(loss.item())
                        validation_loss_history.append(valid_loss.item())

                        plt.plot(counter, training_loss_history, color="blue")
                        plt.plot(counter, validation_loss_history, color="orange")
                        plt.grid()
                        plt.savefig("lcurve_dynamic.svg", format="svg", dpi=1200)

                loss.backward()
                optimizer.step()

            # dynamic learning rate
            if epoch % Config.dynamic_lr == 0 and epoch != 0:
                Config.learning_rate /= 2
                print("leanring rate adjusted to: {}".format(Config.learning_rate))
                # also save the model
                saveSiameseNetwork(net, epoch)
                print("model saved!\n")

    end = time.time()
    print("learning time: ", end - start, "s")

    torch.save(net, "models/model.pt")

    plt.close()

    plt.figure(1)
    plt.plot(counter, training_loss_history, label="training", color="blue")
    plt.plot(counter, validation_loss_history, label="validation", color="orange")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig("lcurve.svg", format="svg", dpi=1200)
    plt.show()


def main():
    Config.run_type = "train"
    train_dataloader, valid_dataloader = load_dataset()
    train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
