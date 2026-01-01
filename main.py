from ultralytics import YOLO
import cv2
import numpy as np
import time

# 1. Load model and move to MPS
model = YOLO("yolo11n-seg.pt").to("mps")

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)

    # 2. Filter for person
    results = model.predict(img, imgsz=640, device="mps", classes=[0], verbose=False)

    for r in results:
        if r.masks is not None:
            for mask_points in r.masks.xy:

                polygon = mask_points.astype(np.int32)
                cv2.drawContours(img, [polygon], -1, (0, 0, 255), 3)


                #TODO add a custom background...




    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Original Feed", img)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()