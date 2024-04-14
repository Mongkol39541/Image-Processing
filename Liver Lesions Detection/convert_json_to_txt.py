import json
import cv2
import pandas as pd
import numpy as np

df = pd.DataFrame([])
imag = []
class_num = []

def convert_labels_to_yolo_format(json_file, output_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for image_filename, annotations in data.items():

        try:
            txt_filename = image_filename.replace('.jpg', '.txt')
            txt_path = f"{output_folder}/{txt_filename}"

            filename = f"images/{image_filename}"
            image = cv2.imread(f'data/train/{image_filename}')
            image_height, image_width, _ = image.shape
            cv2.imwrite(filename, image)
            imag.append(image_filename)
            class_name = 4
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
                class_num.append(class_name)
        except:
            pass

    df['imag'] = imag
    df['class_num'] = class_num
    df.to_csv('data.csv', index=False)

if __name__ == "__main__":
    json_file_path = "data/train.json"
    output_folder_path = "labels"
    convert_labels_to_yolo_format(json_file_path, output_folder_path)