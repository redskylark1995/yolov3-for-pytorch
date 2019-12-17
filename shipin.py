#播放视频
import cv2
import torch
from nms import *
from detector import Detector  as Detector
import PIL.Image as Image
ANCHORS_GROUP = {
        13:[[116,90],[156,198],[373,326]],
        26:[[30,61],[62,45],[59,119]],
        52:[[10,13],[16,30],[33,23]]}

def make_square(im, min_size=416, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2),int((size + x) / 2), int((size + y) / 2)))
    return new_im


path = r"clip1.avi"
cap = cv2.VideoCapture(path)#打开视频文件
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


fps = cap.get(cv2.CAP_PROP_FPS)#FPS帧数，帧数指每一秒有多少张图片
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
out = cv2.VideoWriter('7.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
count = 0
i=0
boxes=""
import torchvision.transforms as T
toimg= T.ToPILImage()
detector = Detector()

transforms = T.Compose([
    T.ToTensor()
])
while True:
    ret,fram = cap.read()#这里ret指的是一种标识，把视频看成是图片的集合，相当于是不断的连续取图片，然后再以一定的时间间隔将图片显示出来。能取到图片ret就是True，然后以50ms的间隔输出图片内容fram，就是视频，当ret取不到东西了，while循环终止
    print("afjlkafsdjkl")
    if ret:#放完了ret就变成了Flase
        _image = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        image = fram
        _image= toimg(_image)
        _image = make_square(_image)

        _img_data = make_square(_image).resize((416,416))

        img_data = transforms(_img_data).reshape([1,3,416,416])
        # image.show()
        # print(image.size)

        y = detector(img_data, 0.3, ANCHORS_GROUP)
        
        y=nms(y.detach().numpy())

        
        for i in range(y.shape[0]):
            box= y[i]
            x1 = int((box[1]-box[3]/2)*1280/416 )
            y1 = int((box[2]-box[4]/2)*1280/416 - (1280-720)/2)
            x2 = int((box[1]+box[3]/2)*1280/416)
            y2 = int((box[2]+box[4]/2)*1280/416 - (1280-720)/2)
            # rec =
            # ,,,
            # print(box[4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.imshow("show",fram)#opencv用imshow显示出来
            #"honey"给当前视频文件一个名称，fram指图片内容，是numpy数组
        # cv2.waitKey(1)#等待时间，毫秒
        out.write(fram)
        count += 1
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break


cap.release()#释放cap视频文件
cv2.destroyAllWindows()#清除所有窗口的东西