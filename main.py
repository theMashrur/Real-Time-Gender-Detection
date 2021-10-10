import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

blue = (255, 0, 0)
red = (0, 0, 255)

#opencv variable to help detect faces, which we can then passed to model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#load model
#by default we're loading in the model in the keras h5 format
classifier = keras.models.load_model('./gender_classifier.h5')

classes_dict = {0 : "man", 1 : "woman"}

def resize_image(image, x, y, w, h):
    '''
    Uses co-ordinates passed as arguments,
    to crop and resize images to tensors of size (1, 100, 100, 3)
    '''
    if x - 0.5*w > 0:
        start_x = int(x - 0.5*w)
    else:
        start_x = x
    if y - 0.5*h > 0:
        start_y = int(y - 0.5*h)
    else:
        start_y = y

    end_x = int(x + (1 + 0.5)*w)
    end_y = int(y + (1 + 0.5)*h)

    face = image[start_y:end_y, start_x:end_x]
    face = tf.image.resize(face, [100, 100])
    face = np.expand_dims(face, axis=0)
    return face

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Cannot open camera")
else:
    print("Camera access granted")

print("To quit, press Ctrl+C")
while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Can't get frame: Exiting")
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            grey,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            image = resize_image(frame, x, y, w, h)
            arr = classifier.predict(image)
            prediction = classes_dict[np.argmax(arr)]
            confidence = round(np.max(arr)*100)
            if prediction == "man":
                colour = blue
            else:
                colour = red
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(frame, "{0}: {1}%".format(prediction, confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)


        cv2.waitKey(1)
        cv2.imshow("Gender Detector", frame)
    except KeyboardInterrupt:
        break

video.release()
cv2.destroyAllWindows()
print("Finished streaming")