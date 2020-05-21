import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pickle

device = None


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def print_network(net):
    print(net)
    print("Number of learnable parameters:", count_parameters(net))
    print("PARAMETERS OF NETWORK")
    i = 0
    weights = []
    biases = []
    for name, param in net.named_parameters():
        if i == 1 or i == 3 or i == 5:
            b = param.detach().numpy().tolist()
            biases.append(b)
        if i == 0 or i == 2 or i == 4:
            w = param.detach().numpy().tolist()
            weights.append(w)
        i += 1
    return weights, biases


def load_network():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load("models/model_100x100.pt").to(device)
    return net


def load_images(id=7):
    transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )
    img0 = Image.open("data/faces/testing/s" + str(id) + "/1.pgm")
    img1 = Image.open("data/faces/testing/s" + str(id) + "/2.pgm")
    img0 = img0.convert("L")
    img1 = img1.convert("L")
    img0 = transform(img0)
    img1 = transform(img1)
    img0, img1 = img0.to(device), img1.to(device)
    img0, img1 = img0[None, ...], img1[None, ...]
    return img0, img1


def show_feature_map(img):
    row, col = 4, 8
    i = 1
    fig = plt.figure(figsize=(row * 3, col * 3))
    fig.add_subplot(row, col, i)
    plt.imshow(img[0][0], interpolation="nearest", cmap="gray")

    activations = net.cnn1[0].forward(img)  # conv2d
    activations = net.cnn1[1].forward(activations)  # maxpooling
    activations = net.cnn1[2].forward(activations)  # relu
    i = 9
    for fmap in activations.detach().numpy()[0]:
        fig.add_subplot(row, col, i)
        plt.imshow(fmap, interpolation="nearest", cmap="inferno")
        i += 1

    activations = net.cnn1[3].forward(activations)
    activations = net.cnn1[4].forward(activations)
    activations = net.cnn1[5].forward(activations)
    i = 17
    for fmap in activations.detach().numpy()[0]:
        fig.add_subplot(row, col, i)
        plt.imshow(fmap, interpolation="nearest", cmap="inferno")
        i += 1

    activations = net.cnn1[6].forward(activations)
    activations = net.cnn1[7].forward(activations)
    activations = net.cnn1[8].forward(activations)
    i = 25
    for fmap in activations.detach().numpy()[0]:
        fig.add_subplot(row, col, i)
        plt.imshow(fmap, interpolation="nearest", cmap="inferno")
        i += 1
    return fig


def save_activations():
    for id in range(7, 20):
        img, _ = load_images(id=id)
        fig = show_feature_map(img)
        plt.tight_layout()
        plt.savefig(
            "data/feature_maps/feature_map" + str(id) + ".png",
            bbox_inches="tight",
            dpi=100,
        )


weights, biases = print_network(load_network())

with open('weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

with open('biases.pkl', 'wb') as f:
    pickle.dump(biases, f)
