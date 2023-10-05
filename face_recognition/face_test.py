import cv2
import os
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
    hog_feature = extract_hog_features(np.array([image]))
    predicted_label = model.predict(hog_feature)[0]
    person_name = label_to_name.get(predicted_label, "Unknown")
    return person_name

def detect_face(test_image_path):
    # Example usage for face recognition
    # test_image_path = 'path/to/test/image.jpg'
    test_image = cv2.imread(test_image_path)
    recognized_person = recognize_face(test_image, model, label_to_name)
    print(f"Recognized person: {recognized_person}")



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
    detect_face("/root/Brad/brad.jpg")


if __name__ == "__main__":
    main()