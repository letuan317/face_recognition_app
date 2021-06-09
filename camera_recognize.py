import face_recognition
import cv2
import numpy as np
import pickle
import os
import setting
import func

from sklearn import neighbors, svm


def write_to_images(name, load_img):
    tss = 0
    while True:
        new_file_name = 'data/images/'+name+"_" + \
            func.get_timestamp()+"_"+str(tss)+'.jpg'
        if not os.path.isfile(new_file_name):
            try:
                cv2.imwrite(new_file_name, load_img)
            except Exception as e:
                print(e)
            break
        else:
            tss += 1


func.start()

print("[INFO] loading encodings...")
data = pickle.loads(open(setting.global_dataset_faces, "rb").read())

print("[INFO] loading trained_knn_model...")
model_path = "data/trained_knn_model.clf"
with open(model_path, 'rb') as f:
    knn_clf = pickle.load(f)

print("[INFO] loading trained_svm_model...")
clf = svm.SVC(gamma='scale')
clf.fit(data["encodings"], data["names"])


video_capture = cv2.VideoCapture(0)


def make_1080p():
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)


def make_720p():
    video_capture.set(3, 1280)
    video_capture.set(4, 720)


def make_480p():
    video_capture.set(3, 640)
    video_capture.set(4, 480)


def change_res(width, height):
    video_capture.set(3, width)
    video_capture.set(4, height)


make_1080p()


#video_capture = cv2.VideoCapture('sample02.mp4')


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_size_scale = 0.25
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.5
previous_face_locations = []
list_unknown_face_encodings = []
list_known_face_encodings = []
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(
        frame, (0, 0), fx=frame_size_scale, fy=frame_size_scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        face_predictions = []
        total_predictions = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                data["encodings"], face_encoding, tolerance=setting.global_tolerance)

            name = "Unknown"
            name_knn = "Unknown"
            name_svm = "Unknown"
            predict_percent = 0
            predict_percent_knn = 0
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]
                # print(face_distances[best_match_index])
                predict_percent = round(
                    100 - face_distances[best_match_index] * 100, 2)
                # print(name, predict_percent)
                # print('[!]origin: '+str(name)+' '+str(predict_percent))

            # knn predict
            distance_threshold = 0.6
            closest_distances = knn_clf.kneighbors(
                face_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <=
                           distance_threshold for i in range(len(face_locations))]
            percent = closest_distances[0][0][0]
            predict_percent_knn = round(100 - percent*100, 2)
            if(predict_percent_knn >= 60):
                for temp_name, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches):
                    name_knn = temp_name
            else:
                predict_percent_knn = 0
            # print('[!]knn: '+str(name)+' '+str(predict_percent_knn))

            # svm predict
            name_svm = clf.predict([face_encoding])[0]

            total_predictions.append(
                [name, predict_percent, name_knn, predict_percent_knn, name_svm])
            # compare names
            if(name == name_knn or name == name_svm):
                name = name
                predict_percent = predict_percent
            elif(name_knn == name_svm and name == "Unknown"):
                name = name_knn
                predict_percent = predict_percent_knn
            elif(name_knn != name_svm and name == "Unknown"):
                name = name_svm
                predict_percent = 0

            face_names.append(name)
            face_predictions.append(predict_percent)

    process_this_frame = not process_this_frame

    if(len(face_locations) > 0):
        # print(len(face_locations))
        for (top, right, bottom, left), encoding, name, predict_percent, total_prediction in zip(face_locations, face_encodings, face_names, face_predictions, total_predictions):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if(name != "Unknown"):
                text_output = name + '  {}%'.format(str(predict_percent))
                color_outline = (52, 235, 70)
            else:
                text_output = name
                color_outline = (0, 0, 255)

            height, width, tee = frame.shape
            expand = 50
            crop_top = top-2*expand
            crop_bottom = bottom+2*expand
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

            # print(crop_top, crop_bottom, crop_left, crop_right)
            # Draw a box around the face
            cv2.rectangle(frame, (crop_left, crop_top),
                          (crop_right, crop_bottom), color_outline, 2)

            # Crop face
            crop_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
            # Draw a label with a name below the face
            cv2.rectangle(frame, (crop_left, crop_bottom), (crop_right,
                                                            crop_bottom-int(font_size*40)), color_outline, cv2.FILLED)

            cv2.putText(frame, str(total_prediction[0]) + " " + str(
                total_prediction[1]), (crop_left + 6, crop_top + 20), font, font_size, color_outline, 1)
            cv2.putText(frame, str(total_prediction[2]) + " " + str(
                total_prediction[3]), (crop_left + 6, crop_top + 40), font, font_size, color_outline, 1)
            cv2.putText(frame, str(
                total_prediction[4]), (crop_left + 6, crop_top + 60), font, font_size, color_outline, 1)

            cv2.putText(frame, text_output, (crop_left + 6,
                                             crop_bottom - 6), font, font_size, (255, 255, 255), 1)

            if(name == "Unknown"):
                unknown_face_encoding = encoding
                if(len(list_unknown_face_encodings) >= 2):
                    unknown_matches = face_recognition.compare_faces(
                        list_unknown_face_encodings, unknown_face_encoding, tolerance=setting.global_tolerance)
                    unknown_face_distances = face_recognition.face_distance(
                        list_unknown_face_encodings, unknown_face_encoding)
                    unknown_best_match_index = np.argmin(
                        unknown_face_distances)
                    if(len(list_unknown_face_encodings) >= 100):
                        list_unknown_face_encodings.pop()
                    if not unknown_matches[unknown_best_match_index]:
                        list_unknown_face_encodings.pop()
                        list_unknown_face_encodings.append(
                            unknown_face_encoding)
                        write_to_images(name, crop_frame)
                else:
                    list_unknown_face_encodings.append(unknown_face_encoding)
                    write_to_images(name, crop_frame)
            else:
                known_face_encoding = encoding
                if(len(list_known_face_encodings) >= 2):
                    known_matches = face_recognition.compare_faces(
                        list_known_face_encodings, known_face_encoding, tolerance=setting.global_tolerance)
                    known_face_distances = face_recognition.face_distance(
                        list_known_face_encodings, known_face_encoding)
                    known_best_match_index = np.argmin(
                        known_face_distances)
                    if(len(list_known_face_encodings) >= 100):
                        list_known_face_encodings.pop()
                    if not known_matches[known_best_match_index]:
                        list_known_face_encodings.append(
                            known_face_encoding)
                        write_to_images(name, crop_frame)
                else:
                    list_known_face_encodings.append(known_face_encoding)
                    write_to_images(name, crop_frame)

        previous_face_locations = face_locations

    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
