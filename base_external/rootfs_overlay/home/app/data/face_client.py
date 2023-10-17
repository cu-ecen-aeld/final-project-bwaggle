import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import time
import sys
import time
import socket


HOST = "192.168.86.33"
PORT = 9000

MODEL_PATH='/data/face_model.joblib'
LABEL_PATH='/data/label_to_name.joblib'

DATASET_DIR = '/data/'

# Extract HOG features
def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    hog_features = []

    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog.compute(gray_image)
        hog_features.append(hog_feature.flatten())

    return np.array(hog_features)

# Write detected photo to file
def write_image_to_volume(person_name, image):
    # Determine the path to the data volume
    data_volume_path = "/data/photos"

    # Create the directory if it doesn't exist
    os.makedirs(data_volume_path, exist_ok=True)

    # Get the current Unix timestamp
    timestamp = int(time.time())

    # Save the image to the data volume
    image_filename = f"{person_name}_{timestamp}.jpg"
    image_path = os.path.join(data_volume_path, image_filename)
    cv2.imwrite(image_path, image)

# Function to establish a socket connection
def connect_to_socket(host, port, max_retries=10000000, retry_interval=1):
    retries = 0
    connected = False
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Waiting for socket to connect.")
    sys.stdout.flush()

    while not connected and retries < max_retries:
        try:
            client_socket.connect((host, port))
            connected = True
        except Exception as e:
            # print(f"Error: {e}")
            retries += 1
            time.sleep(retry_interval)

    if not connected:
        print(f"Failed to establish a connection after {max_retries} retries.")
        sys.stdout.flush()
        return None

    print(f"Connected to socket server {host}:{port}")
    sys.stdout.flush()
    return client_socket

# Function to disconnect the socket
def disconnect_socket(client_socket):
    if client_socket:
        client_socket.close()

# Function to send an image over a connected socket
def send_image(client_socket, image):
    if not client_socket:
        print("Socket is not connected.")
        return

    # Check if the socket is still connected
    try:
        client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    except (socket.error, OSError):
        print("Socket is no longer connected.")
        sys.stdout.flush()
        client_socket = connect_to_socket(HOST, PORT)
        return client_socket

    try:
        # Encode the image as JPEG (you can use other formats like PNG)
        _, img_encoded = cv2.imencode('.jpg', image)

        # Convert the encoded image to bytes
        image_bytes = img_encoded.tobytes()

        # Append an End-Of-File
        image_bytes += b'EOF\n'

        # Send the image size (as an unsigned 64-bit integer)
        image_size = len(image_bytes)
        print(f"Sending image of length {image_size}")
        # client_socket.sendall(image_size.to_bytes(8, byteorder='big'))

        # Send the image data
        try:
            client_socket.sendall(image_bytes)
        except Exception as e:
            print("Socket is no longer connected.")
            sys.stdout.flush()
            disconnect_socket(client_socket)
            client_socket = connect_to_socket(HOST, PORT)
            return client_socket
        finally:
            return client_socket
    except Exception as e:
        print(f"Error: {e}")


# Real-time video stream and face detection
def detect_faces_realtime(model, label_to_name):

    camera_index = 0
    if HOST:
        cap = cv2.VideoCapture(camera_index)
    else:
        gst_str = 'v4l2src device=/dev/video0 ! videoconvert ! appsink'
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    print("Capturing video on camera index: ", camera_index)

    # Connect to the socket
    client_socket = connect_to_socket(HOST, PORT)

    label_ctr_dict = {}

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Ensure there's a frame
        if frame is None:
            break

        gray_frame = frame

        # Resize the frame for processing
        # frame = cv2.resize(frame, (640, 480))

        # Perform face detection using OpenCV's pre-trained haarcascades classifier
        face_cascade = cv2.CascadeClassifier(DATASET_DIR + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        person_name = "Unknown"
        # Draw rectangles around detected faces and label them
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            hog_feature = extract_hog_features(np.array([cv2.resize(face_roi, (128, 128))]))
            hog_feature = hog_feature.reshape(1, -1)
            predicted_label = model.predict(hog_feature)[0]
            person_name = label_to_name.get(predicted_label, "Unknown")
            print(f"Detected person {person_name}")
            

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        if client_socket:
            # Call the function to send the image
            client_socket = send_image(client_socket, frame)
        else:
            # Write image file if same person detected more than 3 times
            if(person_name != "Unknown"):
                if person_name not in label_ctr_dict:
                    label_ctr_dict[person_name] = 0
                label_ctr_dict[person_name] += 1
                print(f"Detected {person_name}:{label_ctr_dict[person_name]}")
                if label_ctr_dict[person_name] > 3:
                    label_ctr_dict[person_name] = 0
                    write_image_to_volume(person_name, frame)
                    print("Logging photo for: ", person_name)
                    sys.stdout.flush()

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    disconnect_socket(client_socket)

def main():

    # Load face detection model
    model = joblib.load(MODEL_PATH)

    # Load model labels (names of person to detect)
    label_to_name = joblib.load(LABEL_PATH)

    # Run in a continuous loop and write photo
    # to file if a person's face is detected
    detect_faces_realtime(model, label_to_name)


if __name__ == "__main__":
    main()