import cv2
import numpy as np
from synergy3DMM import SynergyNet
model = SynergyNet()
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    cv2.rectangle(img, (200, 120), (440, 360), (0, 255, 0), 2)
    img[:480, :200, :] = np.full((480, 200, 3), 0)
    img[:480, 440:, :] = np.full((480, 200, 3), 0)
    img[:120, 200:440, :] = np.full((120, 240, 3), 0)
    img[360:, 200:440, :] = np.full((120, 240, 3), 0)
    roi_img = img[120:360, 200:440]

    # get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
    lmk3d, mesh, pose = model.get_all_outputs(roi_img)
    # cv2.putText(img, "Pitch is: {}".format(Down),
    #             (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    # cv2.putText(img, "Yaw is: {}".format(Shake),
    #             (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    print(lmk3d)
# get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
