#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200131                                 
# REVISED DATE: 20200206 (After peer review)
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    results_dic = dict()

    # Retrieve the filenames from folder image_dir (by default: pet_images/)
    filename_list = listdir(image_dir)
    
    for f in filename_list:
        # TODO: Ignore the systemic files starting with "."
        if f[0] != ".":
            word_list_per_image = f.lower().split("_")

            # TECHNIQUE 1 #
            #pet_labels = ""
            #for word in word_list_per_image:
            #    if word.isalpha():
            #        pet_labels += word + " "
            #pet_labels = pet_labels.strip()
            #print("Filename: '{}'. Pet labels : '{}'".format(f,pet_labels))
            # END TECHNIQUE 1 #

            # TECHNIQUE 2 : comprehension list #
            pet_labels = " " .join([word for word in word_list_per_image if word.isalpha()])
            #for word in word_list_per_image:
            #    if word.isalpha():
            #        pet_labels += word + " "
            #pet_labels = pet_labels.strip()
            #print("Filename: '{}'. Pet labels : '{}'".format(f,pet_labels))
            # END TECHNIQUE 2 #
            
            # Now that we have a good bag of words, we need to put it in the dictionnary
            if f not in results_dic:
                results_dic[f] = [pet_labels] # Convert the string into a list !!!
                #print("New image: adding {} key and {} value".format(results_dic[f], pet_labels))
            #else:
            #    print("Warning: this image has already been added to the dictionary. Image ignored!")
        #else:            
        #    print("Ignoring file {} as it is starting with '.'".format(f))
    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dic
