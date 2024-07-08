import os
import random
from shutil import copyfile
from os import listdir
from os.path import join, isfile, splitext, basename
from PIL import Image
from random import randint
import numpy as np
import cv2
import imageio
import copy

training_ratio = 0.7 # this ratio will be used to generate the mapping files, which will be used for CNTK models later on
root_dir = "cropped_images" # root_dir/source_dir will be the directory with the image patches
dirs = ["0-Not Acne", "1-Clear", "2-Almost Clear", "3-Mild", "4-Moderate", "5-Severe"] #The subdirectory names have to be consistent

source_dir = "Landmarks_Frontal_Faces_Selected_fh_cheeks_patches"                                                                                       #with the image label names in database
dest_dir = "0926_Landmarks_Frontal_Faces_Selected_fh_cheeks_patches" # root_dir/dest_dir/dirs[i] will be the destination
                                                                     # directory for rolled images belonging to the ith label
image_label_file_name = "the3_images_labels.csv" #image label csv file name. Assuming it is in root_dir


mapping_train = os.path.join(root_dir, dest_dir, "mapping_train.txt") #mapping file of the training images
mapping_valid = os.path.join(root_dir, dest_dir, "mapping_valid.txt") #mapping file of the validation images
train_fp = open(mapping_train, 'w')
valid_fp = open(mapping_valid, 'w')
for dir in dirs: #create directories for classes of image patches if not existing
    path = os.path.join(root_dir, dest_dir, dir)
    if not os.path.exists(path):
        os.makedirs(path)

        
imageFiles = [f for f in listdir(join(root_dir, source_dir)) if isfile(join(root_dir, source_dir, f))]
print("There are %d files in the source dir %s"%(len(imageFiles), join(root_dir,source_dir)))

def find_index_of_images(imageFiles, imagename):
    num_images = len(imageFiles)
    index = [i for i in range(num_images) if imagename in imageFiles[i]]
    return index

label_result_file = join(root_dir, image_label_file_name) # Assuming that the image label file is in the root_dir
fp = open(label_result_file, 'r')
fp.readline() # skip the headerline
label_count = {}
max_count = 0

# There is a bug in this handling. We are counting the number of images in each class, from the image label file. 
# However, the image patches we are rolling and allocating are from the selected images. The distribution of the 
# classes in image patches is different from the distribution in the labeled images, which including all non-golden set images.
# That is the reason why after rolling and balancing, the classes of images are still not balanced. 
# Correcting this should have positive impact on the model performance. 
# For now, let's keep as it is. But we need to fix this later on when we retrain the model. 
for row in fp: #Read the count of images in the image label file, and get the number of images of the dominating class
    row = row.strip().split(",")
    label = row[1]
    label_count[label] = label_count.get(label, 0) + 1
    if max_count < label_count[label]: # Get the count of images in the dominating class
        max_count = label_count[label]
fp.close()
print(label_count) 

fp = open(label_result_file, 'r') # Read the image label file again for allocating purpose
fp.readline()
random.seed(98052) # Set the random seed in order to reproduce the result. 

# This is the function that rolls an image patch, and saves the rolled image patch as a jpg file on the destination directory
# img: image data frame before rolling
# dest_path: destination directory to save the rolled image patch
# file_name_wo_ext: file name without extension, i.e., just the image ID
# image_names: a list of image names and path. The new image name and path will be appended to it and returns as an output
# x_or_y: 'x' or 'y'. It specifies whether it is rolling the images right to left, or bottom to top.
# pixels: number of pixels to roll in the direction specified by x_or_y.
# returns: image_names
def roll_and_save(img, dest_path, file_name_wo_ext, image_names, x_or_y, pixels):
    img_height, img_width = img.shape[0:2]
    img2 = copy.copy(img)
    if x_or_y == 'x':
        img2[:, 0:(img_width-pixels),:] = img[:,pixels:img_width,:]
        img2[:,(img_width-pixels):img_width,:] = img[:,0:pixels,:]
    else:
        img2[0:(img_height-pixels), :, :] = img[pixels:img_height, :, :]
        img2[(img_height-pixels):img_height, :,:] = img[0:pixels,:, :]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)        
    dest = join(dest_path, file_name_wo_ext+"_roll_"+x_or_y+"_"+str(pixels)+".jpg") #rolled image file name e.g., 0001_roll_x_112.jpg
    imageio.imwrite(dest, img2) 
    image_names.append(dest)
    return image_names

minimal_roll_times = 2 #Even the dominating class images need to roll twice

for row in fp: # go over for each row in the image label file
    rn = random.uniform(0, 1) # a random number determining whether this file goes to training or validation
    row = row.strip().split(",")
    file_name = row[0] 
    label = row[1]
    file_name_wo_ext = splitext(file_name)[0] #get the image ID
    index = find_index_of_images(imageFiles, file_name_wo_ext) #find the image patches that belong to this image ID
    num_files_found = len(index) #number of image patches found. If the image is not in the selected image set, num_files_found=0
    image_names = []
    for i in range(num_files_found):
        source = join(root_dir, imageFiles[index[i]])
        image_name_no_ext = splitext(imageFiles[index[i]])[0] #get the image patch name, e.g., 0001_fh, 0003_lc, etc.
        img = cv2.imread(source)
        img_height, img_width = img.shape[0:2]
        if 'fh' in imageFiles[index[i]]: # forehead image patches, rolling right to left
            x_or_y = 'x'
        else: #if cheeks, or chins, rolling bottom to top
            x_or_y = 'y'
        roll_ratio = float(max_count)/float(label_count[label]) # determining how many times to roll, in order to balance
        dest_path = join(root_dir, dest_dir, label) #destination path at the image class level
        
        image_names = roll_and_save(img, dest_path, image_name_no_ext, image_names, x_or_y, 0) #save the image without rolling 
        if roll_ratio > 1: # of this is not the dominating class, we need to roll in order to balance
            num_times = int(np.floor(roll_ratio) - 1)
        else:
            num_times = 0
        num_times += minimal_roll_times # adding the number of times that the dominating class is also rolling. 
        if num_times > 0: # determining the step size based on number of times to roll. We want the constant step size for each image
            if x_or_y == 'x':
                step_size = int(np.floor(np.float64(img_width)/np.float64(num_times+1))) 
            else:
                step_size = int(np.floor(np.float64(img_height)/np.float64(num_times+1)))
            for j in range(num_times):
                image_names = roll_and_save(img, dest_path, image_name_no_ext, image_names, x_or_y, step_size*(j+1))
        # The following lines of writing image names to the mapping file have some problem. The image path and name list image_names 
        # is accumulating over the image patches of the same image ID. However, the following lines is writing to the mapping file for
        # every image patch. There will be duplicates in the mapping files. 
        # However, it does not affect the tensorflow models we built since tensorflow was not using the mapping file. 
        # It should not affect the CNTK models either since CNTK models were not using the mapping files I created. 
        # A simple fix of this is to move the following lines to the outer for loop
        label_index = [i for i,x in enumerate(dirs) if x == label][0] # Determining the label index. dirs has 0-Not Acne in the list. 
                                                                      # So, for 1-Clear images, the label index in dirs is 1.
        if label_index >= 1: # We do not model 0-Not Acne, where label_index = 0
            label_index -= 1 #writing the image path and names of the entire rolled image
                             #set for a skin patch to the training mapping file
            if rn <= training_ratio:
                for image_name in image_names:
                  if not train_fp.closed: # Check if train_fp is still open before writing
                    train_fp.write("%s\t%d\n"%(image_name, label_index)) 
            else:
                for image_name in image_names:
                  if not valid_fp.closed: # Check if valid_fp is still open before writing
                    valid_fp.write("%s\t%d\n"%(image_name, label_index))

fp.close()
train_fp.close()
valid_fp.close()
