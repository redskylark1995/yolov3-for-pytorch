from model.darknet import *
from PIL import Image
import PIL.ImageDraw as Draw
import torchvision
import os


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = MainNet()
        if os.path.exists(r"netParameter/"+"net1.pth"):
            self.net.load_state_dict(torch.load(r"netParameter/"+"net1.pth"))
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        mask = output[..., 0] > thresh
        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        idxs = idxs
        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])*416/1280
        h = anchors[a, 1] * torch.exp(vecs[:, 4])*416/1280
        tclass = torch.Tensor([])
        if vecs.shape[0] >0:
            tclass = torch.argmax(vecs[:,5:],dim=1)
        return torch.stack([n.float(), cx, cy, w, h,tclass.float()], dim=1)

ANCHORS_GROUP = {
        13:[[116,90],[156,198],[373,326]],
        26:[[30,61],[62,45],[59,119]],
        52:[[10,13],[16,30],[33,23]]}
        
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def make_square(im, min_size=416, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2),int((size + x) / 2), int((size + y) / 2)))
    return new_im

if __name__ == '__main__':
    _img_data = Image.open(r"img\1.jpg")#os.path.join(r"D:\dataset_data","001.jpg"))
    _img_data = make_square(_img_data).resize((416,416))

    img_data = transforms(_img_data).reshape([1,3,416,416])
    detector = Detector()
    y = detector(img_data, 0.3, ANCHORS_GROUP)
    draw = Draw.ImageDraw(_img_data)
    
    for i in range(y.shape[0]):
        box=y[i]
        # print(box)1
        draw.rectangle((box[1]-box[3]/2,box[2]-box[4]/2,box[1]+box[3]/2,box[2]+box[4]/2),fill=None,outline="yellow")
    _img_data.show()
    print(y)
