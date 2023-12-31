import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
import joblib
import socket
import pickle
import subprocess
import threading
from PIL import Image, ImageTk
import functools


DATASET_DIR = '/home/bwaggle/final-project/final-project-bwaggle/face_server/'
PHOTOS_DIR  = '/home/bwaggle/final-project/data/photos/'
MODEL_DIR = '/home/bwaggle/final-project/data/'

TEST_IMAGE = 'Brad_1.jpg'
TEST_IMAGE_DIR = "/home/bwaggle/final-project/data/photos/Brad/"
TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE)

MODEL_NAME = "face_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

LABEL_NAME = "label_to_name.joblib"
LABEL_PATH = os.path.join(MODEL_DIR, LABEL_NAME)

REMOTE_PATH = '/home/app/data'
PORT = 22
USERNAME = 'root'
IP_ADDRESS = "127.0.0.1"
LOCAL_TEST=True
SERVER_HOST = '0.0.0.0'  # Listen on all available network interfaces
SERVER_PORT = 9000
PASS_FILE = '/home/bwaggle/final-project/data/pwd.txt'


# Define a global variable for storing the current frame received from the socket
current_frame = None

# Define a flag to control the monitoring thread
monitoring = True

# Define the socket server parameters

# Global variables for server and image display
server_socket = None
# is_monitoring = False

# Global variable for the thread
monitoring_thread = None

def load_and_preprocess_images(dataset_dir):
    images = []
    labels = []

    for label, person_name in enumerate(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))  # Resize to a common size
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    hog_features = []

    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog.compute(gray_image)
        hog_features.append(hog_feature.flatten())

    return np.array(hog_features)


def create_label_mapping(dataset_dir):
    label_to_name = {}

    for label, person_name in enumerate(os.listdir(dataset_dir)):
        label_to_name[label] = person_name

    joblib.dump(label_to_name, LABEL_PATH)

    return label_to_name


def recognize_face(image, model, label_to_name):
    # Resize the input image to match training dimensions
    image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features for the input image
    hog = cv2.HOGDescriptor()
    hog_feature = hog.compute(gray_image)

    # Reshape the extracted feature to match the dimensions used for training
    hog_feature = hog_feature.reshape(1, -1)

    # Predict the label using the trained model
    predicted_label = model.predict(hog_feature)[0]
    

    # Get the person's name from the label_to_name mapping
    person_name = label_to_name.get(predicted_label, "Unknown")

    # Draw the recognized name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, person_name, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    

    return person_name

def write_image_to_volume(person_name, image):
    # Determine the path to the data volume
    data_volume_path = "/data"  # Replace with the actual path to your data volume

    # Create the directory if it doesn't exist
    os.makedirs(data_volume_path, exist_ok=True)

    # Save the image to the data volume
    image_filename = f"{person_name}.jpg"
    image_path = os.path.join(data_volume_path, image_filename)
    cv2.imwrite(image_path, image)

    # Provide instructions for accessing the image
    print(f"Recognized person: {person_name}")
    print(f"Image saved to: {image_path}")
    print("You can access the image from the data volume on the remote machine.")


def detect_face(test_image_path, model, label_to_name):
    # Example usage for face recognition
    # test_image_path = 'path/to/test/image.jpg'
    test_image = cv2.imread(test_image_path)
    recognized_person = recognize_face(test_image, model, label_to_name)
    print(f"Recognized person: {recognized_person}")
    return recognized_person


# Function for real-time video stream and face detection
def detect_faces_realtime():
    camera_index = 0
    label_to_name = create_label_mapping(PHOTOS_DIR)
    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(camera_index)
 
    print("Capturing video on camera index: ", camera_index)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Ensure there's a frame
        if frame is None:
            break

        cv2.putText(frame, "Press 'Q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        gray_frame = frame

        # Resize the frame for processing
        frame = cv2.resize(frame, (640, 480))

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

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check for user input to finish capturing photos (press 'q' to quit)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def crop_face(image, bounding_box):
    x, y, w, h = bounding_box  # Get the bounding box coordinates

    # Ensure the coordinates are within the image dimensions
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop the image to the bounding box
    cropped_face = image[y:y+h, x:x+w]

    return cropped_face

# Function to take photos for training
def take_photos(person_name, status_label):
    # Create a directory to save the photos
    photos_dir = PHOTOS_DIR
    os.makedirs(photos_dir, exist_ok=True)


    # Create a directory for the person's photos
    person_dir = os.path.join(photos_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Create a VideoCapture object to capture video from the webcam (you can adjust the device index)
    cap = cv2.VideoCapture(0)

    # Initialize a counter for photo filenames
    photo_counter = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Display instructions in the OpenCV window
        cv2.putText(frame, "Use spacebar to take a photo and 'Q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        gray_frame = frame

        # Resize the frame for processing
        frame = cv2.resize(frame, (640, 480))

        # Perform face detection using OpenCV's pre-trained haarcascades classifier
        face_cascade = cv2.CascadeClassifier(DATASET_DIR + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and label them
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropped_face = frame[y:y+h, x:x+w]
            # cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       
        # Display the frame
        cv2.imshow('Take Photos', frame)

        # Check for user input to capture a photo (spacebar)
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar key
            # Save the photo with a filename that includes the person's name and a counter
            photo_filename = f"{person_name}_{photo_counter}.jpg"
            photo_path = os.path.join(person_dir, photo_filename)
            cv2.imwrite(photo_path, cropped_face)
            print(f"Photo saved as {photo_filename}")

            # Increment the photo counter
            photo_counter += 1
            status_label.config(text=f"{photo_counter} photos of {person_name} were saved")

        # Check for user input to finish capturing photos (press 'q' to quit)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    # Release the VideoCapture and close OpenCV window
    cap.release()



def create_take_photos_tab(tab_control):
    tab_take_photos = ttk.Frame(tab_control)

    # Create a label and entry for the person's name
    name_label = ttk.Label(tab_take_photos, text="Enter First Name:")
    name_entry = ttk.Entry(tab_take_photos)

    # Create a label to provide instructions
    status_label = ttk.Label(tab_take_photos, text="")

    # Create a button to start taking photos
    start_button = ttk.Button(tab_take_photos, text="Start Taking Photos",
                              command=lambda: take_photos(name_entry.get(), status_label))

    # Layout widgets using grid
    name_label.grid(row=0, column=0, padx=10, pady=10)
    name_entry.grid(row=0, column=1, padx=10, pady=10)
    start_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    # finish_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    return tab_take_photos

def train_model(status_label):
    print("Training model")
    # Load and preprocess images
    images, labels = load_and_preprocess_images(PHOTOS_DIR)

    # Extract HOG features
    hog_features = extract_hog_features(images)

    # Create label-to-name mapping
    label_to_name = create_label_mapping(PHOTOS_DIR)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

    # Train a machine learning model (SVM in this example)
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Testing and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Test
    recognized_person = detect_face(TEST_IMAGE_PATH, model, label_to_name)

    # Save model to file
    joblib.dump(model, MODEL_PATH)

    # Update status label
    status_label.config(text=f"Model Accuracy: {accuracy:.2%}\nTesting on {TEST_IMAGE}\nDetected {recognized_person}")


def create_train_model_tab(tab_control):
    tab_train_model = ttk.Frame(tab_control)

    # Create a label to display success message
    status_label = ttk.Label(tab_train_model, text="")
    status_label.pack(padx=10, pady=10)

    # Create a button to trigger model training
    train_button = ttk.Button(tab_train_model, text="Train", command=lambda: train_model(status_label))
    train_button.pack(padx=10, pady=10)

    return tab_train_model

def create_test_model_realtime_tab(tab_control):
    tab_test_model_realtime = ttk.Frame(tab_control)

    # Create a label to display success message
    status_label = ttk.Label(tab_test_model_realtime, text="")
    status_label.pack(padx=10, pady=10)

    # Create a button to trigger model training
    test_button = ttk.Button(tab_test_model_realtime, text="Test Realtime", 
                             command=lambda: detect_faces_realtime())
    test_button.pack(padx=10, pady=10)

    return tab_test_model_realtime

def send_command_to_socket(command):

    try:
        # Define the server's IP address and port
        if LOCAL_TEST:
            server_ip="localhost"
        else:
            server_ip = "192.168.86.52"
        
        server_port = 9000

        # Create a socket and connect to the server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))

        client_socket.send(command.encode())

        # Close the socket connection
        client_socket.close()
    except Exception as e:
        # Handle any errors
        print(f"Error: {e}")

def scp_file_to_client(status_label, ip_entry, port_entry):

    # Define the paths and connection details
    local_path = MODEL_PATH
    remote_path = REMOTE_PATH
    hostname = ip_entry.get()
    port = port_entry.get()
    username = USERNAME
    IP_ADDRESS = ip_entry.get()

    try:
        # Copy face recognition model with SCP
        p = subprocess.Popen(["sshpass", "-f", PASS_FILE, "scp", local_path, f"{username}@{hostname}:{remote_path}"], bufsize=1024)
        sts = os.waitpid(p.pid, 0)
        status_text = f"File {local_path} copied to {username}@{hostname}:{remote_path}"
        status_label.config(text="Status: Raspberry Pi model successfully updated")

        # Copy label_to_name to client with SCP
        local_path = LABEL_PATH
        p = subprocess.Popen(["sshpass", "-f", PASS_FILE, "scp", local_path, f"{username}@{hostname}:{remote_path}"], bufsize=1024)
        sts = os.waitpid(p.pid, 0)
        status_text = f"File {local_path} copied to {username}@{hostname}:{remote_path}"
        status_label.config(text="Status: Raspberry Pi model successfully updated")
        print(status_text)
    except Exception as e:
        print(f"Error: {e}")
        status_label.config(text=f"Error: {e}")


def create_command_tab(tab_control):
    tab_command = ttk.Frame(tab_control)

    # Status label
    status_label = ttk.Label(tab_command, text="Status: OK")
    status_label.grid(row=5, columnspan=4, padx=10, pady=10)

    # Create labels and textboxes for IP address and port
    ip_label = ttk.Label(tab_command, text="IP Address:")
    ip_entry = ttk.Entry(tab_command)
    port_label = ttk.Label(tab_command, text="Port:")
    port_entry = ttk.Entry(tab_command)

    # Place IP address and port controls on grid
    ip_label.grid(row=0, column=0, padx=10, pady=10)
    ip_entry.grid(row=0, column=1, padx=10, pady=10)
    ip_entry.insert(0, "192.168.86.39")
    port_label.grid(row=1, column=0, padx=10, pady=10)
    port_entry.grid(row=1, column=1, padx=10, pady=10)
    port_entry.insert(0, "9000")

    # Button for copying file to client
    button1 = ttk.Button(tab_command, text="Update facial recogntion model on Rpi client", command=lambda: scp_file_to_client(status_label, ip_entry, port_entry))
    button1.grid(row=0, column=3, padx=10, pady=10)

    return tab_command


# Function to handle client connections and image display
def start_server(image_label, stop_event):
    global server_socket

    if (server_socket == None):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)
    

    while not stop_event.is_set():
        client_socket, client_address = server_socket.accept()
        print(f"Accepted connection from {client_address}")

        while not stop_event.is_set():
            try:
                # Receive image data from the client
                data = b""
                while True:
                    packet = client_socket.recv(550000)
                    # print(f"Received packet")
                    if not packet:
                        break
                    data += packet

                    if b'EOF' in data:
                        # print("Detected end of file")
                        break

                if not data:
                    break

                # Convert received data to an image and display it
                print(f"decoding and displaying image size {len(data)}")
                data = data.replace(b'EOF', b'')
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

                # Resize image
                resized_image = cv2.resize(image, (900,600))

                resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(image)
                resized_image_photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image_rgb))


                # image = ImageTk.PhotoImage(image=image)
                # Display image
                image_label.config(image=resized_image_photo)
                image_label.image = resized_image_photo          
                
            except Exception as e:
                print(f"Error: {e}")
                break

        client_socket.close()

    print("Stop monitoring")
    
    if (server_socket):
        server_socket.close()
    


# Function to start the server in a new thread
def start_monitoring(image_label, stop_event):
    global monitoring_thread, server_socket
    if (server_socket):
        server_socket.close
    stop_event.clear()
    monitoring_thread = threading.Thread(target=start_server, args=(image_label, stop_event))
    monitoring_thread.start()

# Function to stop the monitoring thread
def stop_monitoring(stop_event):
    global monitoring_thread, server_socket
    server_socket = None
    print(f"Time to stop monitoring {monitoring_thread}")
    if monitoring_thread:
        print(f"Stopping monitoring_thread {monitoring_thread}")
        if (server_socket):
            server_socket.close
        stop_event.set()


def create_monitor_tab(tab_control, stop_event):
    tab_monitor = ttk.Frame(tab_control)

    # Create a label to display images
    image_label = ttk.Label(tab_monitor)

    start_button = ttk.Button(tab_monitor, text="Monitor Start",
                                command=lambda: start_monitoring(image_label, stop_event))
    stop_button = ttk.Button(tab_monitor, text="Monitor Stop",
                                command=lambda: stop_monitoring(stop_event))
    start_button.pack(padx=10, pady=10)
    stop_button.pack(padx=10, pady=10)
    image_label.pack(padx=10, pady=10)

    return tab_monitor

def on_closing(app, stop_event):
    if monitoring_thread:
        stop_event.set()
    app.destroy()
    

def main():

    # Create the main application window
    app = tk.Tk()
    app.title("Face Recognition Server")

    # Create tabs to separate functionality
    tab_control = ttk.Notebook(app)

    # Create and configure the "Take Photos" tab
    tab_take_photos = create_take_photos_tab(tab_control)

    # Create and configure the "Train Model" tab
    tab_train_model = create_train_model_tab(tab_control)

    # Create and configure the "Test Model Realtime" tab
    tab_test_model_realtime = create_test_model_realtime_tab(tab_control)

    # Create and configure the "Send Command" tab
    tab_command = create_command_tab(tab_control)

    # Create monitoring tab in a seperate thread
    stop_event = threading.Event()
    app.protocol("WM_DELETE_WINDOW", lambda: on_closing(app, stop_event))
    tab_monitor = create_monitor_tab(tab_control, stop_event)

    # Add tabs to the tab control
    tab_control.add(tab_take_photos, text="Take Photos")
    tab_control.add(tab_train_model, text="Train Model")
    tab_control.add(tab_test_model_realtime, text="Test Model Realtime")
    tab_control.add(tab_command, text="Update Client Model")
    tab_control.add(tab_monitor, text="Monitor Client")

    # Place the tab control in the window
    tab_control.pack(expand=1, fill="both")  # Ensure the tab control expands to fill the window

    # Start the main loop for the GUI application
    app.mainloop()


if __name__ == "__main__":
    main()