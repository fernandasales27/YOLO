import cv2
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = model(img_rgb)

    
    rendered = results.render()[0]

    
    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

    
    cv2.imshow('YOLOv5 - Detecção', rendered_bgr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
