import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt


# user defined imports
from model import *
from helper_functions import *
from config import Config
from dataset_helper import *
from loss_function import *

dpi = 100
mpl.rcParams['figure.dpi'] = dpi
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def load_dataset():
    if Config.MNIST:
        folder_dataset_test = dset.MNIST(root='./data', train=False, download=True)
        siamese_dataset_test = MnistDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          transforms.RandomRotation(Config.max_rotation),
                                                                          transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                                                                          transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                                                                          transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
                                                                          transforms.ToTensor(),
                                                                          ])
                                           ,should_invert=False)
    
    elif Config.LFW:
        folder_dataset_test = dset.ImageFolder(root='./data/lfw/faces/testing/')
        siamese_dataset_test = LFWDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                              #transforms.RandomRotation(Config.max_rotation),
                                                                              #transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                                                                              #transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                                                                              #transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
                                                                              #transforms.RandomAffine(0, translate=(0.2,0.2)),
                                                                              transforms.ToTensor(),
                                                                              ])
                                           ,should_invert=False)

    else:
        folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transforms.Compose([transforms.Resize((100,100)),
                                                                              transforms.RandomRotation(Config.max_rotation),
                                                                              #transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                                                                              #transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                                                                              #transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
                                                                              #transforms.RandomAffine(0, translate=(0.2,0.2)),
                                                                              transforms.ToTensor(),
                                                                              ])
                                               ,should_invert=False)


    test_dataloader = DataLoader(siamese_dataset_test,
                                 batch_size=1,
                                 shuffle=True)
    return test_dataloader

def test(net, test_dataloader, y_true, y_pred_raw, y_pred):
    net.eval()
    with torch.no_grad():
        if isinstance(net, (SiameseNetwork, SiameseResNet)):
            y_true_ft = np.array([])
            for test_data in test_dataloader:
                # training data
                img0, img1, label = test_data
                y_true_ft = np.append(y_true_ft, label.item())
                y_true = np.append(y_true, label.item())
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                output1, output2 = net(img0,img1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                y_pred_raw = np.append(y_pred_raw, euclidean_distance.item())
            
            if Config.full_test:
                precisions = np.array([])
                recalls = np.array([])
                thresholds = np.arange(Config.full_test_threshold_start, Config.full_test_threshold_end, Config.full_test_threshold_step)
                for threshold in thresholds:
                    y_pred_ft = np.array([])
                    for prediction in y_pred_raw:
                        if prediction < threshold:  # predict same person
                            prediction = 0
                        else:
                            prediction = 1
                        y_pred_ft = np.append(y_pred_ft, prediction)
                    precision, recall, _, _ = precision_recall_fscore_support(y_true_ft, y_pred_ft, average='weighted')
        
                    precisions = np.append(precisions, precision)
                    recalls = np.append(recalls, recall)
             
            for prediction in y_pred_raw:
                if prediction < Config.threshold:  # predict same person
                    prediction = 0
                else:
                    prediction = 1
                y_pred = np.append(y_pred, prediction)
    
            return y_true, y_pred, precisions, recalls, thresholds

        elif isinstance(net, (SiameseNetworkAbs, SiameseResNetConcat)):
            for test_data in test_dataloader:
                # training data
                img0, img1 , label = test_data
                y_true = np.append(y_true, label.item())
                img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
                output = net(img0,img1)
                probability = torch.sigmoid(output)
                y_pred_raw = np.append(y_pred_raw, probability.item())
            
            for prediction in y_pred_raw:
                if prediction < 0.5:  # predict same person
                    prediction = 0
                else:
                    prediction = 1
                y_pred = np.append(y_pred, prediction)
            return y_true, y_pred


def print_results(y_true, y_pred, fscores, precisions = None, recalls = None, thresholds = None):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(y_pred)):
        pred = y_pred[i]
        true = y_true[i]
        if pred == true == 1:
            tn += 1
        elif pred == true == 0:
            tp += 1
        elif pred == 0 and true == 1:
            fp += 1
        elif pred == 1 and true == 0:
            fn += 1

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    fscores.append(fscore)

    print("-------------------------")
    print("accuracy_score:", round(accuracy, 4))
    print("precision:     ", round(precision, 4))
    print("recall:        ", round(recall, 4))
    print("fscore:        ", round(fscore, 4))
    print("-------------------------")
    print("total true positives", tp)
    print("total true negatives", tn)
    print("-------------------------")
    print("total false positives:", fp)
    print("total false negatives:", fn)
    print("-------------------------")
    try:
        if thresholds.any() != None:
            precision_recall_curve_plotter(precisions, recalls, thresholds)
    except AttributeError:
        pass
    

def precision_recall_curve_plotter(precisions, recalls, thresholds):
    if Config.full_test:
        plt.plot(thresholds, recalls)
        plt.plot(thresholds, precisions)
        leg = plt.legend(('precision', 'recall'), frameon=True)
        leg.get_frame().set_edgecolor('k')
        plt.xlabel('threshold')
        plt.ylabel('rate')
        plt.show()


def visual_test(net):
    # Visualization
    if Config.MNIST:
        folder_dataset_test = dset.MNIST(root='./data', train=False, download=True)
        siamese_dataset = MnistDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          #transforms.RandomRotation(Config.max_rotation),
                                                                          #transforms.RandomVerticalFlip(p=Config.p_flip),
                                                                          #transforms.RandomHorizontalFlip(p=Config.p_flip),
                                                                          #transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
                                                                          transforms.ToTensor(),
                                                                          ])
                                           ,should_invert=False)
    elif Config.LFW:
        folder_dataset_test = dset.ImageFolder(root='./data/lfw/faces/testing/')
        siamese_dataset = LFWDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                              #transforms.RandomRotation(Config.max_rotation),
                                                                              #transforms.RandomVerticalFlip(p=Config.p_ver_flip),
                                                                              #transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
                                                                              #transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
                                                                              #transforms.RandomAffine(0, translate=(0.2,0.2)),
                                                                              transforms.ToTensor(),
                                                                              ])
                                           ,should_invert=False)
    else:
        folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transforms.Compose([transforms.Resize((100,100)),
#                                                                  transforms.RandomRotation(Config.max_rotation),
#                                                                  transforms.RandomVerticalFlip(p=Config.p_ver_flip),
#                                                                  transforms.RandomHorizontalFlip(p=Config.p_hor_flip),
#                                                                  transforms.RandomResizedCrop(100, scale=(0.9,1), ratio=(1,1)),
#                                                                  transforms.RandomAffine(0, translate=(0.60,0.60)),
                                                                  transforms.ToTensor(),
                                                                  ])
                                               ,should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,num_workers=0,shuffle=True)
    dataiter = iter(test_dataloader)
    x0,_,label1 = next(dataiter)

    net.eval()
    torch.no_grad()
    if isinstance(net, (SiameseNetwork,SiameseResNet)):
        for i in range(10):
            try:
                _,x1,label1 = next(dataiter)
            except StopIteration:
                break
            concatenated = torch.cat((x0,x1),0)
            output1, output2 = net(x0.to(device), x1.to(device))
            distance = F.pairwise_distance(output1, output2)

            imshow(torchvision.utils.make_grid(concatenated),'Score: {:.2f}'.format(distance.item()))
            #imshow(torchvision.utils.make_grid(concatenated))

    elif isinstance(net, SiameseNetworkAbs):
        for i in range(0):
            _,x1,_ = next(dataiter)
            concatenated = torch.cat((x0,x1),0)
            output = net(x0.to(device), x1.to(device))
            score = torch.sigmoid(output)

            imshow(torchvision.utils.make_grid(concatenated),'Score: {:.2f}'.format(score.item()))
            

def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    Config.run_type = "test"

    fscores = []
    for file in os.listdir("models"):
        if file.endswith(".pt"):
            print("testing with model:", os.path.join(file))
            net = torch.load("models/" + file).to(device)

            y_true = np.array([])
            y_pred_raw = np.array([])
            y_pred = np.array([])
            y_true_total = np.array([])
            y_pred_total = np.array([])
            best_thresholds = np.array([])
            
            precisions_total = np.arange(Config.full_test_threshold_start, Config.full_test_threshold_end, Config.full_test_threshold_step)
            precisions_total.fill(0)
            
            recalls_total = np.arange(Config.full_test_threshold_start, Config.full_test_threshold_end, Config.full_test_threshold_step)
            recalls_total.fill(0)
            
            with tqdm(total=Config.number_of_tests) as pbar:
                for i in range(Config.number_of_tests):
                    test_dataloader = load_dataset()
                    
                    if isinstance(net, (SiameseNetworkAbs, SiameseResNetConcat)):
                        y_true, y_pred = test(net, test_dataloader, y_true, y_pred_raw, y_pred)
                        
                        y_true_total = np.append(y_true_total, y_true)
                        y_pred_total = np.append(y_pred_total, y_pred)
                        pbar.update(1)
                        
                    else:
                        y_true, y_pred, precisions, recalls, thresholds = test(net, test_dataloader, y_true, y_pred_raw, y_pred)

                        y_true_total = np.append(y_true_total, y_true)
                        y_pred_total = np.append(y_pred_total, y_pred)
                        pbar.update(1)

                        for j in range(len(thresholds)):
                            precisions_total[j] += precisions[j]
                            recalls_total[j] += recalls[j]
                
                    precisions_mean = np.true_divide(precisions_total, Config.number_of_tests)
                    recalls_mean = np.true_divide(recalls_total, Config.number_of_tests)
                
            
            if isinstance(net, (SiameseNetworkAbs, SiameseResNetConcat)):
                print_results(y_true_total, y_pred_total, fscores, precisions_mean, recalls_mean)
            else:
                print_results(y_true_total, y_pred_total, fscores, precisions_mean, recalls_mean, thresholds)
                maxPrecisions = np.argmax(precisions_mean)
                best_thresholds = np.append(best_thresholds, thresholds[maxPrecisions])
            print("\n\n")

            visual_test(net)
            if isinstance(net, (SiameseNetwork,SiameseResNet)):
                best_threshold = np.mean(best_thresholds)
                print("Best threshold value: ", best_threshold)
    
    plt.plot(fscores)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.show()
    #plt.savefig("fscoreplot.svg",  format='svg', dpi=1200)
    
    #print(fscores)

if __name__ == '__main__':
    main()