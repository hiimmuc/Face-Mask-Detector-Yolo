import os
import time
from typing import Any

import cv2
import numpy as np
from imutils.video import FPS


class Yolov4:
    def __init__(self, net_path, config, label) -> None:
        self.net_path = net_path
        self.config = config
        self.label = label
        self.old_indices = []
        self.old_boxes = []
        self.old_id = []
        self.old_conf = []
        self.net = self.create_net(self.config, self.net_path)

    def detector(self, image, confidence_threshold, nms_threshold, delay=0) -> Any:
        # image = cv2.resize(image, None, fx=0.5, fy=0.5,
        #                    interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255., (416, 416), [0, 0, 0], swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_outputs = self.net.forward(output_layers)

        # english ver
        # class_names = []
        # with open(self.label, "r") as f:
        #     class_names = [line.strip() for line in f.readlines()]

        # viet ver
        class_names = ["Deo chua dung", "Da deo", "Chua deo"]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    centerX, centerY, width, height = list(
                        map(int, detection[0:4] * [w, h, w, h]))

                    top_leftX, top_leftY = int(
                        centerX - width / 2), int(centerY - height / 2)
                    width, height = int(width), int(height)

                    boxes.append([top_leftX, top_leftY, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, nms_threshold)
        self.num_obj = len(indices)

        if delay <= 2 and self.num_obj == 0:
            indices = self.old_indices
            boxes = self.old_boxes
            class_ids = self.old_id
            confidences = self.old_conf

        list_coor = []
        new_box = []
        crop_scale = 0.1
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x = abs(int(x - crop_scale * w / 2))
                y = abs(int(y - crop_scale * h / 2))
                w = abs(int((1 + 2 * crop_scale) * w))
                h = abs(int((1 + 0 / 2) * h))  # can change 0 to cropscale
                new_box.append((x, y, w, h))

        # list_coor = sorted(list_coor, key=lambda x: x[0])

        # # draw boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.masked, self.unmasked = 0, 0
        for i in range(len(new_box)):
            if i in indices:
                x, y, w, h = new_box[i]
                tag = f"{class_names[class_ids[i]]}:{int(confidences[i]*100)}%"
                color = colors[class_ids[i]]
                if class_ids[i] == 1:
                    self.masked += 1
                elif class_ids[i] == 2:
                    self.unmasked += 1

                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
                cv2.putText(image, tag, (x, y - 5), font, 0.6,
                            color, 1, lineType=cv2.LINE_AA)
                list_coor.append([x, y, w, h])

        self.old_indices = indices
        self.old_boxes = boxes
        self.old_id = class_ids
        self.old_conf = confidences
        return image, [self.num_obj, self.masked, self.unmasked]

    def create_net(self, config, net_path):
        net = cv2.dnn.readNetFromDarknet(config, net_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[INFO] Done reading net!")
        return net


# label = r"backup/obj.names"
# config = r"backup/yolov4-tiny-custom.cfg"
# net_path = r"backup/yolov4-tiny-custom_best.weights"


# print("[INFO] Loading net...")
# t = time.time()
# myYolo = Yolov4(net_path=net_path, config=config, label=label)
# print(f"[INFO] Done in {round(time.time() - t, 2)} s")

# # test image


# def test_image(path):
#     test_img = myYolo.load_image(path)
#     t = time.time()
#     coor, output_img = myYolo.detector(test_img, 0.7, 0.3)
#     print(coor)
#     print(time.time() - t, "s")

#     cv2.imshow("res", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def test_video(path=0):
#     # test video
#     cap = cv2.VideoCapture(0)
#     fps = FPS().start()
#     width, height = list(map(int, [cap.get(3), cap.get(4)]))
#     delay = 0
#     while cap.isOpened():
#         _, frame = cap.read()
#         h, w, _ = frame.shape
#         output_img, cond = myYolo.detector(frame, 0.3, 0.5, delay)
#         if cond:
#             delay = delay + 1 if delay <= 3 else 0
#         output_img = cv2.resize(output_img, (640, 480))
#         cv2.imshow("result", output_img)
#         fps.update()
#         if cv2.waitKey(1) == 27:
#             break

#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     test_video()
