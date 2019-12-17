import numpy as np
import PIL.ImageDraw as draw
import PIL.Image as Img
import matplotlib.pyplot as plt

def datas(num):
    list = [[]]
        
    for i1 in range(num):       
        h = np.random.randint(5,12)
        w = np.random.randint(5,12)
        x1 = np.random.randint(0,80-h)
        y1 = np.random.randint(0,80-w)
        x2 = h+x1
        y2 = w+y1

        zxd = np.random.randint(0,100)
        list.append([x1,y1,x2,y2,zxd])
               
    point1 = np.array(list)
    return point1
def IOU(a,isMin=False):
    x1 = a[:,1]-a[:,3]/2
    y1 = a[:,2]-a[:,4]/2
    x2 = a[:,1]+a[:,3]/2
    y2 = a[:,2]+a[:,4]/2
    score =  (x2[0]-x1[0])*(y2[0]-y1[0])
    scores = (x2[1:]-x1[1:])*(y2[1:]-y1[1:])
        
    xx1 = np.maximum(x1[0],x1[1:])
    yy1 = np.maximum(y1[0],y1[1:])
    xx2 = np.minimum(x2[0],x2[1:])
    yy2 = np.minimum(y2[0],y2[1:])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    scores2 = w*h

    if isMin :
        return scores2/np.minimum(score,scores)
    else:
        return scores2/(score+scores-scores2)
        
def nms(point1,ismin = False):
    '''
    根据iou去掉框
    ismin表示时候使用最小iou
    '''
    after = point1.shape[0]
    # print(point1.shape)
    if point1.shape[0]<1:
        point1=point1[point1[:,0].argsort()[::-1]]

    pointout = np.array([])
    i=0
    while point1.shape[0] >0:
        
        # print("未处理目标框数量:",point1.size/5 )
        if pointout.size == 0:
            
            pointout= point1[0].reshape((1,-1))
        else:
            
            pointout= np.vstack((pointout,point1[0].reshape((1,-1))))

            
        i+=1
        index = np.where(IOU(point1,ismin)<=0.3)[0]+1
        
        point1=point1[index]
    # print("after:",after,"\tbefore:",pointout.shape[0])
    return pointout


# pointout = nms(datas(0))
# img = Img.new("RGB",(225,225),(255,255,255))
# img_draw = draw.ImageDraw(img)
# for i in range(pointout.shape[0]):
    
    
#     img_draw.rectangle((pointout[i,0],pointout[i,1],pointout[i,2],pointout[i,3]),outline="blue")
# plt.imshow(img)
# plt.show() 