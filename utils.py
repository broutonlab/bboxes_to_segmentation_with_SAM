import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


class SamConverter:
    def __init__(self, sam_weights, sam_type, device, eps=0.01):
        sam = sam_model_registry[sam_type](checkpoint=sam_weights)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.eps = eps
        self.h, self.w = None, None

    def set_image(self, image):
        self.predictor.set_image(image)
        self.h, self.w = image.shape[:2]

    def box2mask(self, cxywh):
        cls = int(cxywh[0])
        cx, cy = int(cxywh[1] * self.w), int(cxywh[2] * self.h)
        bw, bh = int(cxywh[3] * self.w), int(cxywh[4] * self.h)

        input_box = np.array([cx-bw//2, cy-bh//2, cx+bw//2, cy+bh//2])
        input_point = np.array([[cx, cy]])
        input_label = np.array([1])
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        return cls, masks[0]

    def mask2segm(self, cls, mask):
        thresh = cv2.threshold(mask.astype(float), 0.5, 255, 0)[1].astype(np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, self.eps * peri, True).astype(float)
        approx = approx[:, 0, :]
        approx[:, 0] /= self.w
        approx[:, 1] /= self.h
        approx = approx.flatten().tolist()
        return [cls, *approx]

    def box2segm(self, cxywh):
        cls, mask = self.box2mask(cxywh)
        return self.mask2segm(cls, mask)
