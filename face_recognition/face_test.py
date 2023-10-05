import cv2
import os
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_DIR = '/root/'

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

    # Print dimensions for debugging
    print(f"Shape of hog_feature: {hog_feature.shape}")
    print(f"Shape of training data: {model.support_vectors_.shape[1]}")

    # Predict the label using the trained model
    predicted_label = model.predict(hog_feature)[0]

    # Get the person's name from the label_to_name mapping
    person_name = label_to_name.get(predicted_label, "Unknown")

    # Draw the recognized name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, person_name, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the image with the recognized name
    cv2.imshow("Recognized Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return person_name





def detect_face(test_image_path, model, label_to_name):
    # Example usage for face recognition
    # test_image_path = 'path/to/test/image.jpg'
    test_image = cv2.imread(test_image_path)
    recognized_person = recognize_face(test_image, model, label_to_name)
    print(f"Recognized person: {recognized_person}")
    cv2.imshow(test_image_path)



def is_ubuntu():
    try:
        # Run the grep command to check for "ubuntu" in /etc/os-release
        result = subprocess.run(['grep', '-q', 'ubuntu', '/etc/os-release'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check the return code to see if "ubuntu" was found
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print("Error:", e)
        return False


def main():
    # Detect environment
    if is_ubuntu():
        print("Running on Ubuntu")
        DATASET_DIR = '/home/bwaggle/images/'
    else:
        print("Not running on Ubuntu")

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
    detect_face("/home/bwaggle/images/Vaughn/vaughn2.jpg", model, label_to_name)


if __name__ == "__main__":
    main()