import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

HOST = False

if HOST:
    DATASET_DIR = '/home/bwaggle/final-project/final-project-bwaggle/base_external/rootfs_overlay/home/app/data/'
    TEST_IMAGE_PATH = "/home/bwaggle/final-project/final-project-bwaggle/base_external/rootfs_overlay/home/app/data/Brad/brad1.jpg"
else:
    DATASET_DIR = '/data/'
    TEST_IMAGE_PATH = "/data/Brad/brad1.jpg"



# Step 3: Load and preprocess images
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

# Step 4: Extract HOG features
def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    hog_features = []

    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog.compute(gray_image)
        hog_features.append(hog_feature.flatten())

    return np.array(hog_features)

# Step 5: Assign labels to names
def create_label_mapping(dataset_dir):
    label_to_name = {}

    for label, person_name in enumerate(os.listdir(dataset_dir)):
        label_to_name[label] = person_name

    return label_to_name



# Step 12: Face recognition
import tempfile
import os

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
    
    if not HOST:
        write_image_to_volume(person_name, image)

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

# Function for real-time video stream and face detection
def detect_faces_realtime(model, label_to_name):
    # cv2.namedWindow("Window", cv2.WINDOW_X11)
    # Create a VideoCapture object to capture video from the webcam (you can adjust the device index)
    camera_index = 0
    if HOST:
        cap = cv2.VideoCapture(camera_index)
    else:
        gst_str = 'v4l2src device=/dev/video0 ! videoconvert ! appsink'
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    print("Capturing video on camera index: ", camera_index)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Ensure there's a frame
        if frame is None:
            break

        # Convert frame to grayscale if it's not already in grayscale
        # if len(frame.shape) == 3:
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_frame = frame

        gray_frame = frame

        # Resize the frame for processing
        frame = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale for face detection
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        if(person_name != "Unknown"):
            print("Found live person: ", person_name)
            break

        # Display the frame with detected faces
        # if HOST:
        cv2.imshow('Face Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    if not HOST:
        write_image_to_volume(person_name, frame)

def main():
    # Step 6: Load and preprocess images
    images, labels = load_and_preprocess_images(DATASET_DIR)

    # Step 7: Extract HOG features
    hog_features = extract_hog_features(images)

    # Step 8: Create label-to-name mapping
    label_to_name = create_label_mapping(DATASET_DIR)

    # Step 9: Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

    # Step 10: Train a machine learning model (SVM in this example)
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Step 11: Testing and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Test
    detect_face(TEST_IMAGE_PATH, model, label_to_name)

    # Detect faces in real-time video stream
    detect_faces_realtime(model, label_to_name)

if __name__ == "__main__":
    main()