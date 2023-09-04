import os

import argparse
import cv2

from utils import SamConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Videos to images")
    parser.add_argument('dir', type=str, help='Input dataset dir')
    parser.add_argument('--device', type=str, default="cuda", help='Device : cuda or cpu')
    parser.add_argument('--weights', type=str, default="sam_vit_h_4b8939.pth", help='Checkpoint for SAM')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of SAM')
    args = parser.parse_args()

    sam = SamConverter(args.weights, args.model_type, args.device)

    labels_path = os.path.join(args.dir, 'labels')
    images_path = os.path.join(args.dir, 'images')

    segmentations_path = os.path.join(args.dir, 'segmentation_labels')
    if not os.path.exists(segmentations_path):
        os.makedirs(segmentations_path)

    for label_name in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_name)) as f:
            labels = f.readlines()
        image_name = f"{label_name.split('.')[0]}.jpg"
        image = cv2.imread(os.path.join(images_path, image_name))
        sam.set_image(image)
        labels = [list(map(lambda x: float(x), label.strip("\n").split(" "))) for label in labels]
        segmentation_labels = []
        for label in labels:
            segmentation_labels.append(sam.box2segm(label))
        segmentation_str = "\n".join(" ".join(str(l) for l in label) for label in segmentation_labels)
        with open(os.path.join(segmentations_path, label_name), 'w') as f:
            f.write(segmentation_str)
