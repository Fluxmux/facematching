import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

device = None


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def print_network(net):
    print(net)
    print("Number of learnable parameters:", count_parameters(net))
    print("PARAMETERS OF NETWORK")
    i = 0
    for param in net.parameters():
        if i == 0:
            for x in range(0, 3):
                for y in range(0, 3):
                    print(param[0, 0, x, y].item())
        i += 1


def load_network():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load("models/model_e100.pt").to(device)
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


net = load_network()
# img0, img1 = load_images()
print_network(net)
# out0, out1 = net(img0, img1)
# dist = F.pairwise_distance(out0, out1).item()
#
"""
img, _ = load_images()
activations = net.cnn1[0].forward(img)  # conv2d
out = activations.detach().numpy()
out = out[0, 0].flatten()
print(np.where(out == -0.218048095703125))
print(out)
"""
"""
activations = net.cnn1[1].forward(activations)  # maxpooling
activations = net.cnn1[2].forward(activations)  # relu
"""
