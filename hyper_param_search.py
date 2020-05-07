import torch.nn as nn
import torch
from torch import optim
from config import Config
from sklearn.utils.estimator_checks import check_estimator
from loss_function import ContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self, in_layers = 1,
        layers_1 = 16,
        layers_2 = 32,
        layers_3 = 64,
        out_features = 128,
        kernel_size_1 = 7,
        kernel_size_2 = 5,
        kernel_size_3 = 3,
        intermediary_features = 512,
        lr = Config.learning_rate,
        wd = Config.weight_decay):
        
        super(SiameseNetwork, self).__init__()
        self.in_layers = in_layers
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.layers_3 = layers_3
        self.out_features = out_features
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3
        self.intermediary_features = intermediary_features
        self.lr = lr
        self.wd = wd
        
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.in_layers, self.layers_1, kernel_size=self.kernel_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.layers_1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.layers_1, self.layers_2, kernel_size=self.kernel_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.layers_2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),


            nn.ReflectionPad2d(1),
            nn.Conv2d(self.layers_2, self.layers_3, kernel_size=self.kernel_size_3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.layers_3),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        )
        

        self.fc1 = nn.Sequential(
            nn.Linear(self.layers_3*11*11, self.intermediary_features),
            nn.ReLU(inplace=True),

            nn.Linear(self.intermediary_features, self.out_features),
        )
        
        self.optimizer = optim.Adam(self.get_params(), lr = self.lr, weight_decay = self.wd)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"in_layers": self.in_layers,
                "layers_1": self.layers_1,
                "layers_2": self.layers_2,
                "layers_3": self.layers_3,
                "out_features": self.out_features,
                "kernel_size_1": self.kernel_size_1,
                "kernel_size_2": self.kernel_size_2,
                "kernel_size_3": self.kernel_size_3,
                "intermediary_features": self.intermediary_features,
                "lr": self.lr,
                "wd": self.wd}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, data):
        net = self.to(device)
        net.train()
        img0, img1 , label = data
        img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
        self.optimizer.zero_grad()
        output1, output2 = net(img0,img1)
        loss = ContrastiveLoss(output1,output2,label)
        loss.backward()
        self.optimizer.step()
    
    def predict(data):
        pass    

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)   
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def main():   
    
if __name__ == '__main__':
    main()
