import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision
import datetime

# user defined imports
from model import *
from helper_functions import *
from config import *
from dataset_helper import *
from loss_function import *
from yoloface import *

def load_dataset():
    folder_dataset_webcam = dset.ImageFolder(root=Config.webcam_dir)
    siamese_dataset_webcam = SiameseNetworkDataset(imageFolderDataset=folder_dataset_webcam,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                           ,should_invert=False)
    
    
    webcam_dataloader = DataLoader(siamese_dataset_webcam, 
                                 batch_size=1, 
                                 shuffle=True)
    return webcam_dataloader
    
def crop_img(frame, left, top, width, height, filename):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[top:top+height, left:left+width]
    pil = Image.fromarray(crop_gray)
    pil = transforms.functional.resize(pil, 100)
    pil.save(Config.webcam_dir + "s41/" + filename + ".pgm")

def test(net, img0, img1):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    net.eval()
    with torch.no_grad():
        if isinstance(net, (SiameseNetwork,SiameseResNet)):
            output1, output2 = net(img0.to(device), img1.to(device))
            euclidean_distance = F.pairwise_distance(output1, output2)
            concatenated = torch.cat((img0,img1),0)
            imshow(torchvision.utils.make_grid(concatenated),'Score: {:.2f}'.format(euclidean_distance.item()))
            return euclidean_distance.item()
        
        elif isinstance(net, (SiameseNetworkAbs, SiameseResNetConcat)):
            output = net(img0.to(device), img1.to(device))
            probability = torch.sigmoid(output) 
            concatenated = torch.cat((img0,img1),0)
            imshow(torchvision.utils.make_grid(concatenated),'Score: {:.2f}'.format(probability.item()))
            return probability.item()
            

def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    Config.run_type = "webcam"
    
    net = torch.load("models/model.pt").to(device)    # selects neural net
    
    cam = cv2.VideoCapture(0)   # selects camera
    
    if not(cam.isOpened()):
        cam.open()
    
    print("press 'q' to quit")
    print("press 'a' and 'b' to capture image A and B")
    print("press 'c' to compare")
    img0 = img1 = None
    text = None
    color = None
    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame,text,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        cv2.imshow('cam feed', frame)
        
        key = cv2.waitKey(1)
            
        if key & 0xFF == ord('q'):  # close camera
            cam.release()
            cv2.destroyAllWindows() 
            break
        elif key & 0xFF == ord('a'):    # capture single frame
            filename = "1"
#            create_img(frame, filename)
            left, top, width, height = detect_faces(frame)         # excecute face detection and return coordinates
            crop_img(frame, left, top, width, height, filename)
            print("captured", filename)
        elif key & 0xFF == ord('b'):    # capture single frame
            filename = "2"
            left, top, width, height = detect_faces(frame)         # excecute face detection and return coordinates
            crop_img(frame, left, top, width, height, filename)
            print("captured", filename)
        elif key & 0xFF == ord('c'):    # compare the two images
            webcam_dataloader = load_dataset()
            print("testing...")
            for i, train_data in enumerate(webcam_dataloader, 0):
                img0, img1, _ = train_data
                img0 = img0.to(device)
                img1 = img1.to(device)
                score = test(net, img0, img1)
                print("\nscore: " + str(score))
                if isinstance(net, (SiameseNetwork,SiameseResNet)):
                    if score < Config.threshold:
                        text = "Same"
                        color = (0,255,0)
                        print("This is the same person!\n")
                    else:
                        text = "Different"
                        color = (0,0,255)
                        print("This is a different face!\n")
                elif isinstance(net, (SiameseNetworkAbs, SiameseResNetConcat)):
                    if score < 0.5:
                        text = "Same"
                        color = (0,255,0)
                        print("This is the same person!\n")
                    else:
                        text = "Different"
                        color = (0,0,255)
                        print("This is a different face!\n")
                #break
           
if __name__ == '__main__':
    main()
