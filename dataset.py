import torch
from torch.utils.data import Dataset
import os
import torchvision
from PIL import Image
import math
import numpy as np

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

ANCHORS_GROUP = {
   13:[[116,90],[156,198],[373,326]],
   26:[[30,61],[62,45],[59,119]],
   52:[[10,13],[16,30],[33,23]]}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
CLASS_NUM = 10
def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b
    
from PIL import Image

def make_square(im, min_size=416, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2),int((size + x) / 2), int((size + y) / 2)))
    return new_im,size,x, y
class yolov3Dataset(torch.utils.data.Dataset):
    def __init__(self,laberpath,dataPath):
        super(yolov3Dataset, self).__init__()
        with open(laberpath) as f:
            self.dataset = f.readlines()
        self.dataPath = dataPath

    def __getitem__(self, index):
        labels = {}
        line =  self.dataset[index].split()
        _img_data = Image.open(os.path.join(self.dataPath,line[0]))

        _img_data,max_size,old_x,old_y = make_square(_img_data,416)
        _img_data = _img_data.resize((416,416))

        img_data = transforms(_img_data)
        
        _boxes = np.array([float(x) for x in line[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)
        for feature_size, anchors in ANCHORS_GROUP.items():
            labels[feature_size] = torch.zeros(size=(feature_size, feature_size, 3, 5 + 10),dtype=torch.float32)

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset,   = math.modf((cx+(max_size-old_x)/2) * feature_size / max_size)
                cy_offset, cy_index = math.modf((cy +(max_size-old_y)/2)* feature_size / max_size)
                w,h=w*416/1280,h*416/1280
                for i, anchor in enumerate(anchors):
                    anchor_area = ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    
                    labels[feature_size][int(cy_index), int(cx_index), i] = torch.Tensor(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(CLASS_NUM, int(cls))])

        return labels[13], labels[26], labels[52], img_data

    def __len__(self):
        return len(self.dataset)