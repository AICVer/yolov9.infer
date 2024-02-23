import cv2
from PIL import Image
import numpy as np

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)    
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]    
    return img.astype(np.float32)

image_path = 'data/bus.jpg'
# image_path = 'zidane.jpg'

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
inputs = preprocess(img)
print(inputs.shape)

model = cv2.dnn.readNetFromONNX('weights/gelan-c.onnx')
model.setInput(inputs)
output_name = model.getUnconnectedOutLayersNames()
output = model.forward(output_name[0])


pred = np.transpose(output[0], (1, 0))

bboxes = pred[:, :4]
scores, clases = np.max(pred[:, 4:], axis=1), np.argmax(pred[:, 4:], axis=1)

# cxcywh --> xywh
bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2

conf_thres = 0.25
iou_thres = 0.45
index = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), conf_thres, iou_thres)


img_np = cv2.imread(image_path)
img_np = cv2.resize(img_np, (640, 640))
for bbox, cls, score in zip(bboxes[index], clases[index], scores[index]):
    label = f'{int(cls)}, {score:.2f}'
    xywh = [int(i) for i in bbox]    

    cv2.rectangle(img_np, (xywh[0], xywh[1], xywh[2], xywh[3]), (0, 255, 0), 2)
    cv2.putText(img_np, label, (xywh[0], xywh[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow('1', img_np)
cv2.waitKey()
