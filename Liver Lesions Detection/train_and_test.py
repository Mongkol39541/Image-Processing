import pandas as pd
import cv2
import json
import numpy as np

df = pd.read_csv('data.csv')

che_0 = df[df['class_num'] == 0]
che_1 = df[df['class_num'] == 1]
che_2 = df[df['class_num'] == 2]
che_4 = df[df['class_num'] == 4]


num_0 = len(che_0)
train_0 = che_0[:num_0 - int(num_0 * 0.1)]
test_0 = che_0[num_0 - int(num_0 * 0.1):]
num_1 = len(che_1)
train_1 = che_1[:num_1 - int(num_1 * 0.1)]
test_1 = che_1[num_1 - int(num_1 * 0.1):]
num_2 = len(che_2)
train_2 = che_2[:num_2 - int(num_2 * 0.1)]
test_2 = che_2[num_2 - int(num_2 * 0.1):]
num_4 = len(che_4)
train_4 = che_4[:num_4 - int(num_4 * 0.1)]
test_4 = che_4[num_4 - int(num_4 * 0.1):]

def convert_labels_to_yolo_format(json_file, output_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for image_filename, annotations in data.items():
        che = False
        for name in train_0['imag']:
            if image_filename == name:
                che = True
                break
        if che:
            txt_filename = image_filename.replace('.jpg', '.txt')
            txt_path = f"{output_folder}/{txt_filename}"

            filename = f"images_new/{image_filename}"
            image = cv2.imread(f'data/train/{image_filename}')
            image_height, image_width, _ = image.shape
            cv2.imwrite(filename, image)
            with open(txt_path, 'w') as txt_file:
                for i in range(len(annotations['labels'])):
                    class_name = annotations['labels'][i]
                    bbox = annotations['boxes'][i]
                    x_center = (bbox[0] + bbox[2]) / (2.0 * image_width)
                    y_center = (bbox[1] + bbox[3]) / (2.0 *  image_height)
                    bbox_width = (bbox[2] - bbox[0]) / image_width
                    bbox_height = (bbox[3] - bbox[1]) / image_height

                    line = f"{class_name} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    txt_file.write(line)

if __name__ == "__main__":
    json_file_path = "data/train.json"
    output_folder_path = "labels_new"
    convert_labels_to_yolo_format(json_file_path, output_folder_path)