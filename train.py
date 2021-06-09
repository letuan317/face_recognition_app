import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import math
from sklearn import neighbors
import numpy as np
import pickle
from imutils import paths
import os
import sys
import cv2
import calendar
import time

import setting
import func


class trainBot:
    def __init__(self):
        self.folder_train = setting.global_folder_train
        self.folder_people = setting.global_folder_people

    def main(self):
        self.inputNameOfPerson()
        self.load_data()
        self.get_face()
        self.train()

    def inputNameOfPerson(self):
        while(True):
            check = ''
            self.nameOfPerson = input("Name: ")
            self.nameOfCompany = input("Company: ")
            self.name = self.nameOfCompany+'_'+self.nameOfPerson
            self.list_names, filenames = func.get_list_files(
                self.folder_people)
            self.path_name_folder = self.folder_people+self.name+'/'
            func.mkdir(self.path_name_folder)
            func.mkdir(self.path_name_folder+'backup')
            func.mkdir(self.path_name_folder+'new_faces')
            func.mkdir(self.path_name_folder+'train')
            func.mkdir(self.path_name_folder+'trained_faces')
            if (self.name in self.list_names):
                check = input(
                    "Name is existed. Do you want to add more? (y/n) ")
                if(check.lower() == 'y'):
                    print("Copy images for trainning to : " +
                          self.path_name_folder+'train')
                    input("Enter for continue: ")
                    return self.name
            else:
                print("\n[INFO] Copy images for trainning to : " +
                      self.path_name_folder+'train')
                input("Enter for continue: ")
                return self.name

    def load_data(self):
        print("[INFO] loading encodings...")
        self.path_data = self.path_name_folder+self.name+'.dat'
        try:
            data = pickle.loads(
                open(self.path_data, "rb").read())
            self.list_face_encodings = data["encodings"]
            self.list_names = data["names"]
            print(len(self.list_face_encodings), len(self.list_names))
        except Exception as e:
            print(e)
            self.list_face_encodings = []
            self.list_names = []

    def get_face(self):
        print("[INFO] get_face to new_faces...")
        self.list_images = list(paths.list_images(
            self.path_name_folder+'train/'))
        count = 0
        check_get_face = False
        if(len(self.list_images) > 0):
            for img_path in self.list_images:
                print("[+] get_face image {}/{} {}".format(count +
                                                           1, len(self.list_images), img_path))
                checkProcess = True
                try:
                    load_image = face_recognition.load_image_file(img_path)
                except:
                    print('No such file ' + img_path)
                    checkProcess = False

                if(checkProcess == True):
                    face_locations = face_recognition.face_locations(
                        load_image)
                    if(len(face_locations) > 0):
                        print("[INFO] Found {} faces".format(
                            len(face_locations)))
                        for (top, right, bottom, left) in face_locations:
                            img_color = cv2.imread(
                                img_path, cv2.IMREAD_COLOR)
                            height, width, tee = img_color.shape
                            expand = 50
                            crop_top = top-expand
                            crop_bottom = bottom+expand
                            crop_left = left-expand
                            crop_right = right+expand
                            if(crop_top < 0):
                                crop_top = 0
                            if(crop_bottom > height):
                                crop_bottom = height
                            if(crop_left < 0):
                                crop_left = 0
                            if(crop_right > width):
                                crop_right = width

                            crop_img = img_color[crop_top:crop_bottom,
                                                 crop_left:crop_right]
                            height, width, channels = crop_img.shape
                            if(height >= 100 and width >= 100):
                                tss = 0
                                while True:
                                    ts = calendar.timegm(time.gmtime())
                                    new_file_name = self.path_name_folder + \
                                        'new_faces/'+self.name + \
                                        '_'+str(ts)+str(tss)+'.jpg'
                                    if not os.path.isfile(new_file_name):
                                        cv2.imwrite(new_file_name, crop_img)
                                        check_get_face = True
                                        break
                                    else:
                                        tss += 1
                    if(check_get_face == True and len(face_locations) > 0):
                        check_get_face = False
                        filename = os.path.basename(img_path)
                        tss = 0
                        while True:
                            temp_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            ts = calendar.timegm(time.gmtime())
                            new_file_name = self.path_name_folder + \
                                'backup/'+self.name + \
                                '_'+str(ts)+str(tss)+'.jpg'
                            if not os.path.isfile(new_file_name):
                                cv2.imwrite(new_file_name, temp_img)
                                check_get_face = True
                                break
                            else:
                                tss += 1

                        os.remove(img_path)
                    else:
                        if(len(face_locations) == 0):
                            print("[INFO] Found {} faces".format(
                                len(face_locations)))
                        elif(check_get_face == False):
                            print('[!] Face_croped is too small')
                        os.remove(img_path)
                        check_get_face = False
                count += 1
        print("\n[INFO] Check new faces: " +
              self.path_name_folder+'new_faces')
        input("Enter for continue: ")

    def train(self):
        print("[INFO] train...")
        self.list_images = list(paths.list_images(
            self.path_name_folder+'new_faces/'))
        count = 0
        if(len(self.list_images) > 0):
            for img_path in self.list_images:
                print("[+] train image {}/{} {}".format(count +
                                                        1, len(self.list_images), img_path))
                checkProcess = True
                try:
                    load_image = face_recognition.load_image_file(img_path)
                    # print(img)
                except:
                    print('No such file ' + img_path)
                    checkProcess = False

                if(checkProcess == True):

                    face_locations = face_recognition.face_locations(
                        load_image)
                    face_encodings = face_recognition.face_encodings(
                        load_image, face_locations)

                    for face_encoding in face_encodings:
                        if(len(self.list_face_encodings) > 0):
                            matches = face_recognition.compare_faces(
                                self.list_face_encodings, face_encoding, tolerance=setting.global_tolerance)
                            check_name = "Unknown"
                            face_distances = face_recognition.face_distance(
                                self.list_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                check_name = self.list_names[best_match_index]
                                print(face_distances[best_match_index])
                                predict_percent = round(
                                    100 - face_distances[best_match_index] * 100, 2)
                                print(check_name, predict_percent)
                            else:
                                print('[!] Moving file to trained_faces')
                                self.list_face_encodings.append(face_encoding)
                                self.list_names.append(self.name)
                                tss = 0
                                while True:
                                    ts = calendar.timegm(time.gmtime())
                                    new_file_name = self.path_name_folder + \
                                        'trained_faces/'+self.name + \
                                        '_'+str(ts)+str(tss)+'.jpg'
                                    if not os.path.isfile(new_file_name):
                                        temp_file = cv2.imread(
                                            img_path, cv2.IMREAD_COLOR)
                                        cv2.imwrite(new_file_name, temp_file)

                                        break
                                    else:
                                        tss += 1

                                os.remove(img_path)
                        else:
                            print('[!] Init first encoding')
                            self.list_face_encodings.append(face_encoding)
                            self.list_names.append(self.name)
                            tss = 0
                            while True:
                                ts = calendar.timegm(time.gmtime())
                                new_file_name = self.path_name_folder + \
                                    'trained_faces/'+self.name + \
                                    '_'+str(ts)+str(tss)+'.jpg'
                                if not os.path.isfile(new_file_name):
                                    cv2.imwrite(new_file_name, load_image)
                                    break
                                else:
                                    tss += 1
                            os.remove(img_path)
                count += 1

        print(len(self.list_face_encodings), len(self.list_names))
        data = {"encodings": self.list_face_encodings, "names": self.list_names}
        f = open(self.path_data, "wb")
        f.write(pickle.dumps(data))
        f.close()


def data_transfer():
    print("[INFO] Transfering data from people")
    dirnames, filenames = func.get_list_files("data/people")
    path_data = 'data/data.dat'

    list_face_encodings = []
    list_names = []

    for person in dirnames:
        print("[INFO] " + person)
        temp_path_data = "data/people/" + person+'/'+person+'.dat'
        try:
            data = pickle.loads(
                open(temp_path_data, "rb").read())
            temp_list_face_encodings = data["encodings"]
            for encoding in temp_list_face_encodings:
                list_face_encodings.append(encoding)

            temp_list_names = data["names"]
            for name in temp_list_names:
                list_names.append(name)
        except Exception as e:
            print(e)

    data = {"encodings": list_face_encodings, "names": list_names}
    f = open(path_data, "wb")
    f.write(pickle.dumps(data))
    f.close()


def train_knn():
    print("[INFO] Training a k-nearest neighbors classifier for face recognition. ")
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """

    train_dir = "data/people/"
    model_save_path = "data/trained_knn_model.clf"
    n_neighbors = 3
    knn_algo = 'ball_tree'
    verbose = False

    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(os.path.join(train_dir, class_dir), 'trained_faces')):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(
                    image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    bot = trainBot()
    bot.main()
    check = input("Do you want to combine data? (y/n): ")
    if(check.lower() == 'y'):
        data_transfer()
        train_knn()
