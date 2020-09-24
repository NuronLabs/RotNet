from __future__ import print_function

import os
import cv2
import numpy as np
import argparse

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import RotNetDataGenerator, crop_largest_rectangle, angle_error, rotate
import datetime


def process_images(model, input_path, output_path,
                   batch_size=64, log=False):
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']

    output_is_image = False
    if os.path.isfile(input_path):
        image_paths = [input_path]
        if os.path.splitext(output_path)[1].lower() in extensions:
            output_is_image = True
            output_filename = output_path
            output_path = os.path.dirname(output_filename)
    else:
        image_paths = [os.path.join(input_path, f)
                       for f in os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in extensions]
        if os.path.splitext(output_path)[1].lower() in extensions:
            print('Output must be a directory!')

    predictions = model.predict_generator(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        )
    )

    predicted_angles = np.argmax(predictions, axis=1)
    rotated_angles = [] # We will be using this to compile what angles we actually rotate by for logging

    if output_path == '':
        output_path = '.'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path, predicted_angle in zip(image_paths, predicted_angles):
        image = cv2.imread(path)
        isLandscape = image.shape[1] >= image.shape[0] # image.shape = [height, width, channels]
        if 45 <= predicted_angle < 315:
            """
            Angle rotation logic:
            if 45 < pred_angle < 315:
                if isLandscape: rotate by +-90
                else: rotate by 0/180
            """
            if 45 <= predicted_angle and predicted_angle < 135 and not isLandscape:
                angle_cv2 = cv2.ROTATE_90_CLOCKWISE
                rotated_angle = 90
            elif 135 <= predicted_angle and predicted_angle < 225 and isLandscape:
                angle_cv2 = cv2.ROTATE_180
                rotated_angle = 180
            elif 225 <= predicted_angle and predicted_angle < 315 and not isLandscape:
                angle_cv2 = cv2.ROTATE_90_COUNTERCLOCKWISE
                rotated_angle = 270
            else:
                rotated_angles.append(0)
                continue
            rotated_image = cv2.rotate(image, angle_cv2)
            rotated_angles.append(rotated_angle)
            print("Rotated {} {}°".format(path, rotated_angle))
            if not output_is_image:
                output_filename = os.path.join(output_path, os.path.basename(path))
            cv2.imwrite(output_filename, rotated_image)
        else:
            rotated_angles.append(0)
    
    if log:
        """
        Create a CSV file and dump the logs in the format
        "image_filename, rotation_angle"
        """
        with open("log-{}.csv".format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')),'a') as f:
            f.write("Filename, Angle\n")
            for path, rotated_angle in zip(image_paths, rotated_angles):
                f.write("{}, {}\n".format(os.path.basename(path), rotated_angle))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model')
    parser.add_argument('input_path', help='Path to image or directory')
    parser.add_argument('-o', '--output_path', help='Output directory')
    parser.add_argument('-b', '--batch_size',
                        help='Batch size for running the network')
    parser.add_argument('-l', '--log', help='Log for future training')
    args = parser.parse_args()

    print('Loading model...')
    model_location = load_model(args.model, custom_objects={
                                'angle_error': angle_error})
    output_path = args.output_path if args.output_path else args.input_path

    print('Processsing input image(s)...')
    process_images(model_location, args.input_path, output_path,
                   args.batch_size, args.log)
