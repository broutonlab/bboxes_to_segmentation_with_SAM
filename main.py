import os
from tqdm import tqdm
import argparse
import cv2

from utils import SamConverter

OK_EXT = ['jpg', 'png', 'jpeg']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bboxes to segments with SAM.")
    parser.add_argument('dir', type=str, help='Input dataset dir')
    parser.add_argument('--device', type=str, default="cuda", help='Device : cuda or cpu')
    parser.add_argument('--weights', type=str, default="sam_vit_h_4b8939.pth", help='Checkpoint for SAM')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of SAM')
    parser.add_argument('--approx_eps', type=float, default=0.001, help='Approx_eps, use in approximate segment polygon.')
    
    parser.add_argument('--merge_mask', type=bool, default=True, help='Merge mask in each masks.')
    parser.add_argument('--merge_kernel_size', default=[33,33], type=int, nargs='+', help='size kernel for merging.')
    args = parser.parse_args()

    merge_kernel_size = tuple(args.merge_kernel_size)
    
    sam = SamConverter(args.weights, args.model_type, args.device,
                        args.approx_eps, args.merge_mask, args.merge_kernel_size)

    labels_path = os.path.join(args.dir, 'labels')
    images_path = os.path.join(args.dir, 'images')
    
    assert all([item.split('.')[-1] in OK_EXT  for item in os.listdir(images_path)]), \
        f'ERROR: acceptable file extensions: {OK_EXT}'
    assert all([item.split('.')[-1] =='txt' for item in os.listdir(labels_path)]), \
        f'ERROR: labels extensions must be ".txt" - yolo format'
        
    segmentations_path = os.path.join(args.dir, 'segmentation_labels')
    if not os.path.exists(segmentations_path):
        os.makedirs(segmentations_path)

    for image_name in tqdm(os.listdir(images_path)):
        label_name = f"{image_name.split('.')[0]}.txt"
        
        if not os.path.exists(os.path.join(labels_path, label_name)):
            continue

        with open(os.path.join(labels_path, label_name)) as f:
            labels = f.readlines()

        image = cv2.imread(os.path.join(images_path, image_name))
        sam.set_image(image)
        
        labels = [list(map(lambda x: float(x), label.strip("\n").split(" "))) for label in labels]
        segmentation_labels = []
        for label in labels:
            segmentation_labels.append(sam.box2segm(label))
        segmentation_str = "\n".join(" ".join(str(l) for l in label) for label in segmentation_labels)
        with open(os.path.join(segmentations_path, label_name), 'w') as f:
            f.write(segmentation_str)
