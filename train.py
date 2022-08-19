from glmk import *

import argparse

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50,ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    os.makedirs('./models',exist_ok=True)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    extractor = create_feature_extractor(model,[NODE]).to(device)

    for parameters in extractor.parameters():
        parameters.requires_grad = False

    dataset = FingerDataset('./data/train/train_data.txt','./data/train')
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=True,workers=args.workers)
    network = CollabNetwork(shgn_levels=2,shgn_iterations=2).to(device)
    criterion = FingerLoss()
    optimizer = torch.optim.Adam(params=network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    print("Training started.")
    for epoch in range(1,args.epoch+1):
        for idx, data in enumerate(dataloader):
            img,landmarks_gt,heatmap_gt,gesture_index = data
            img = img.to(device)
            landmarks = landmarks.to(device)
            heatmap = heatmap.to(device)
            resnet_features = extractor(img)[NODE]
            prev_gest = prev_pfld = prev_shgn = 0 
            gesture_gt = torch.zeros(3).to(device)
            gesture_gt[:,gesture_index]=1
            
            for _ in range(args.iterations):
                optimizer.zero_grad()
                landmarks,heatmaps,gesture,prev_gest,prev_pfld,prev_shgn = network(resnet_features,prev_gest,prev_pfld,prev_shgn)
                loss = criterion(gesture, landmarks, heatmaps, gesture_gt, landmarks_gt, heatmap_gt)
                loss.backward()
                optimizer.step()
        torch.save(network.state_dict(),'./models/model.pth.tar')        
        print(f"Epoch #{epoch} Batch #{idx} Loss:{loss.item()}")
    print("Training ended.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--workers',help='amount of workers fetching dataset',dest='workers',type=int,default=8)
    parser.add_argument('-l','--lr',help='adjust learning rate',dest='lr',type=float,default=0.001)
    parser.add_argument('-b','--batch_size',help='adjust batch size',dest='batch_size',type=int,default=16)
    parser.add_argument('-E','--epoch',help='adjust training epoch',dest='epoch',type=int,default=32)
    parser.add_argument('-p','--patience',help='adjust scheduler patience',dest='patience',type=int,default=4)
    parser.add_argument('-d','--weight_decay',help='adjust weight decay',dest='weight_decay',type=float,default=1e-6)
    parser.add_argument('-i','--iterations',help='adjust iterations of collab learning',dest='iterations',type=int,default=2)
    main(parser.parse_args())
