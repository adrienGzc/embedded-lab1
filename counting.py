import argparse
import tensorflow as tf
import tensornets as nets
from cv2 import cv2 as cv
import numpy as np
import time

tf.compat.v1.disable_eager_execution()

def main(stream, photo):
    with tf.compat.v1.Session() as sess:
        # Get YOLOV3 model from tensornet and store it.
        inputs = tf.compat.v1.placeholder(tf.float32, [None, 416, 416, 3]) 
        model = nets.YOLOv3COCO(inputs, nets.Darknet19)
        # Choose the class to identify. I'm only need the person class to count people.
        classes = {'0' : 'person'}
        indexOfClasses = [0]
        # Train the model.
        sess.run(model.pretrained())

        # Launch webcam to capture the image from the camera. 0 is for the webcam.
        webcamStream = cv.VideoCapture(photo)
        # Start main loop for the webcam stream.
        while(webcamStream.isOpened()):
            # Get the image of the stream.
            ret, frame = webcamStream.read()

            # Create new image based on the stream shape.
            img = cv.resize(frame, (416, 416))
            imgTmp = np.array(img)
            imgTmp = np.reshape(imgTmp, (-1, 416, 416, 3))

            # Run model with the image of the stream for classification and identifying people.
            preds = sess.run(model.preds, {inputs: model.preprocess(imgTmp)})

            # Create boxes for people in the image.
            boxes = model.get_boxes(preds, imgTmp.shape[1:3])

            # Create a windows to display the image.
            cv.namedWindow('image', cv.WINDOW_NORMAL)
            cv.resizeWindow('image', 500, 500)

            # Loop to create the boxes for people found in image if in it and draw the boxe around.
            # Also display the number of classes identify. If 2 people on camera should display 'Number of person: 2'.
            boxes1 = np.array(boxes)
            for classe in indexOfClasses:
                count = 0
                label = classes[str(classe)]
                if len(boxes1) != 0:
                    for index in range(len(boxes1[classe])): 
                        box = boxes1[classe][index]
                        if boxes1[classe][index][4] >= .40: 
                            count += 1
                            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                            cv.putText(img, label, (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv.LINE_AA)
                print('Number of person: ', count)

            # Display the image.
            cv.imshow("image", img)

            # If you give an image as input, show the image with the result for 5 secondes and stop the program.
            if photo != 0:
                cv.waitKey(5000)
                webcamStream.release()
                cv.destroyAllWindows()
                break

            # If you're in a real-time stream and you press 'q', this will quit stop the program.
            if stream == True:
                if cv.waitKey(1) & 0xFF == ord('q'):
                    webcamStream.release()
                    cv.destroyAllWindows()
                    break

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", help="Input image to count number of people")
    img = vars(args.parse_args())

    stream = False
    if img['image'] == None:
        stream = True
        img['image'] = 0
    main(stream, img['image'])