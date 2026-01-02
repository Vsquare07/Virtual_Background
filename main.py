from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
# 1. Load model and move to GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_HUMANS = 1
print(DEVICE)
model = YOLO("yolo11m-seg.pt").to(DEVICE)

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640,400))
    # 2. Filter for person
    results = model.predict(img, device=DEVICE, classes=[0], verbose=False)
    for r in results:
        boxes = r.boxes
        if r.masks is not None:
            background_frame = cv2.imread("background.jpg")
            background_frame = cv2.resize(background_frame, (640,400))

            for mask_points in r.masks.xy:
                polygon = mask_points.astype(np.int32)
                
                # 1. Create a 1-channel mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                
                # 2. Get the image of human
                res_roi = cv2.bitwise_and(img, img, mask=mask)
                
                # 3. Get the background by flipping the mask
                mask_inv = cv2.bitwise_not(mask)
                res_backgroundred = cv2.bitwise_and(background_frame, background_frame, mask=mask_inv)
                
                # 4. Merge them back into the main image
                img = cv2.add(res_roi, res_backgroundred)
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("hehe", img)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()