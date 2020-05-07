import matplotlib.pyplot as plt
import numpy as np
import torch
# user defined imports
from config import Config

# helper functions
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
def saveSiameseNetwork(model, epoch):
    torch.save(model, "models/model_e"  + str(epoch).zfill(3) + ".pt")
    
def loadSiameseNetwork():
    return torch.load(Config.save_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)