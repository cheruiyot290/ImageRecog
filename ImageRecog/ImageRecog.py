import cv2
import face_recognition
from deepface import DeepFace

def detect_faces(image_path):
    # Load the image using face_recognition
    image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Return face locations
    return face_locations

def draw_faces(image, face_locations):
    # Draw rectangles around the faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

def predict_gender_and_age(image_path, face_location):
    # Load the image for gender and age prediction
    results = DeepFace.analyze(image_path, actions=['gender', 'age'])

    # Check if results is a list
    if isinstance(results, list):
        # Assume the first element of the list contains the result
        first_result = results[0]

        # Extract gender and age predictions
        gender = first_result['gender']
        age = first_result['age']
    else:
        # Results is a dictionary
        gender = results['gender']
        age = results['age']

    # Draw the face on the image
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Print gender and age predictions
    print(f"Gender - {gender}, Age - {age}")

if __name__ == "__main__":
    # Example usage
    image_path = r'C:\Users\cheru\OneDrive\Desktop\elvis.jpg'

    # Detect faces
    faces = detect_faces(image_path)

    # Load the image using OpenCV for display
    image = cv2.imread(image_path)

    # Predict gender and age and draw faces
    for idx, face_location in enumerate(faces):
        predict_gender_and_age(image_path, face_location)

    # Display the image with faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
