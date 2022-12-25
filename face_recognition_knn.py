from calendar import c
import math
from msilib.schema import Directory
from sklearn import neighbors
import os
import os.path
import pickle
# pip install pillow
from PIL import Image, ImageDraw
# pip3 install face_recognition
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

#파일 이동을 위함
import shutil
#폴더 선택창을 위함
import tkinter
from tkinter import filedialog

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # 학습된 데이터로 Classifier를 생성합니다.
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 학습된 KNN classifier를 저장합니다.
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
   
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



#얼굴표시 
def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":

    root = tkinter.Tk()
    root.withdraw()
    train_dir = filedialog.askdirectory(parent=root, initialdir="/",title= "Please select a train directory")
    test_dir = filedialog.askdirectory(parent=root, initialdir="/",title= "Please select a test directory")
    result_dir = filedialog.askdirectory(parent=root, initialdir="/",title= "Please select a result directory")
    
    name = ""
    # STEP 1: 로컬에 저장되어있는 파일로 학습을 시작합니다.
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
   # classifier = train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2:  학습된 classifier로 확인되지 않은 이미지들을 분석합니다.
    for image_file in os.listdir(test_dir):
        full_file_path = os.path.join(test_dir, image_file)

        print("Looking for faces in {}".format(image_file))

        #Classifier로 사진에서 발견된 사랍들을 분석합니다.
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        count = 0
        names  = []
        for name, (top, right, bottom, left) in predictions:
            #발견된 사람마다 이름을 출력하고
            print("- Found {} at ({}, {})".format(name, left, top))
            #발견된 사람의 수를 증가시킵니다.
            count += 1
            names.append(name)

            if count > 1 :
                result = list(set(names))
                if len(result)>1:
                    # 발견된 사람이 많으면 many라는 폴더에 별도로
                    name = "many"
                   
                        
    
        # Display results overlaid on an image
        #show_prediction_labels_on_image(os.path.join(test_dir, image_file), predictions)
 
# Directory create & move file

        if name != "":   
            os.makedirs('{}/{}'.format(result_dir,name),exist_ok=True)
            shutil.move(os.path.join(test_dir,image_file), os.path.join("{}/{}".format(result_dir,name),image_file))
            print("{} is move to result/{}".format(image_file,name))
            name = ""
        else :
            os.makedirs('{}/delete'.format(result_dir),exist_ok=True)
            print ("{} is skip ".format(image_file))
            shutil.move(os.path.join(test_dir,image_file), os.path.join("{}/delete".format(result_dir),image_file))
    print("eof")