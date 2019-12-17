import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from model.darknet import *
import os
torch.nn.Softmax()
lossMSE = nn.MSELoss()
def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0
    # print("index",mask_obj.shape)

    output[mask_obj][0]=F.sigmoid(output[mask_obj][0])
    output[mask_noobj][0]=F.sigmoid(output[mask_noobj][0])
    output[mask_obj][5:]=F.softmax(output[mask_obj][5:],dim=1)

    loss_obj = lossMSE(output[mask_obj],target[mask_obj])
    loss_noobj = lossMSE(output[mask_noobj],target[mask_noobj])
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss

if __name__ == '__main__':

    myDataset = dataset.yolov3Dataset(r"img\label.txt",r"img")
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=4, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = MainNet().to(device)
    
    # if os.path.exists(r"netParameter/"+"net.pth"):
    #     net.load_state_dict(torch.load(r"netParameter/"+"net.pth"))
    net.train()

    opt = torch.optim.Adam(net.parameters())
    i=0
    while True:
        i+=1
        meanloss =0
        for target_13, target_26, target_52, img_data in train_loader:

            target_13, target_26 = target_13.to(device), target_26.to(device)
            target_52, img_data = target_52.to(device), img_data.to(device)

            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss.item())
            meanloss += loss
        
        print (i,"   meanloss",meanloss.item()/8)
        if i%10==0:
            torch.save(net.state_dict(),r"netParameter/"+"net1.pth")
            print("save net")
