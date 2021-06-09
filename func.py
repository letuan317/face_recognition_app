import os
import datetime


def start():
    print("///////////////////////////////////////")
    print("//      __  __       __       _      //")
    print("//     | | / /      /  \     | |     //")
    print("//     | |/ /      / /\ \    | |     //")
    print("//     | | \      / /__\ \   | |     //")
    print("//     | |\ \    / /----\ \  | |     //")
    print("//     |_|  \_\ /_/      \_\ |_|     //")
    print("///////////////////////////////////////")


def mkdir(folderPath):
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)


def addWordToFile(filePath, word):
    fa = open(filePath, 'a')
    fa.write(word)
    fa.close()


def get_list_files(folder_path):
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        return dirnames, filenames


def get_timestamp():
    x = datetime.datetime.now()
    return x.strftime("%Y")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'_'+x.strftime("%H")+'-'+x.strftime("%M")+'-'+x.strftime("%S")


def crop_faces(load_image, face_locations):
    for (top, right, bottom, left) in face_locations:
        height, width, channels = load_image.shape
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

        crop_img = load_image[crop_top:crop_bottom,
                              crop_left:crop_right]
        height, width, channels = crop_img.shape
    return crop_img, height, width
