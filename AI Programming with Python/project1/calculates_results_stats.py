#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#                                                                             
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200202                                 
# REVISED DATE: 20200206 (After peer review)
# PURPOSE: Create a function calculates_results_stats that calculates the 
#          statistics of the results of the programrun using the classifier's model 
#          architecture to classify the images. This function will use the 
#          results in the results dictionary to calculate these statistics. 
#          This function will then put the results statistics in a dictionary
#          (results_stats_dic) that's created and returned by this function.
#          This will allow the user of the program to determine the 'best' 
#          model for classifying the images. The statistics that are calculated
#          will be counts and percentages. Please see "Intro to Python - Project
#          classifying Images - xx Calculating Results" for details on the 
#          how to calculate the counts and percentages for this function.    
#         This function inputs:
#            -The results dictionary as results_dic within calculates_results_stats 
#             function and results for the function call within main.
#         This function creates and returns the Results Statistics Dictionary -
#          results_stats_dic. This dictionary contains the results statistics 
#          (either a percentage or a count) where the key is the statistic's 
#           name (starting with 'pct' for percentage or 'n' for count) and value 
#          is the statistic's value.  This dictionary should contain the 
#          following keys:
#     Z      n_images - number of images
#     A      n_correct_dogs - number of correctly classified dog images
#     B      n_dogs_img - number of dog images
#     C      n_correct_notdogs - number of correctly classified NON-dog images
#     D      n_notdogs_img - number of NON-dog images
#     E      n_correct_breed - number of correctly classified dog breeds
#     Y      n_match - number of matches between pet & classifier labels
#    opt     pct_match - percentage of correct matches
#    1a      pct_correct_dogs - percentage of correctly classified dogs
#     2      pct_correct_breed - percentage of correctly classified dog breeds
#    1b      pct_correct_notdogs - percentage of correctly classified NON-dogs 
#
##
# TODO 5: Define calculates_results_stats function below, please be certain to replace None
#       in the return statement with the results_stats_dic dictionary that you create 
#       with this function
# 
def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using classifier's model 
    architecture to classifying pet images. Then puts the results statistics in a 
    dictionary (results_stats_dic) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value. See comments above
                     and the previous topic Calculating Results in the class for details
                     on how to calculate the counts and statistics.
    """        
    # Replace None with the results_stats_dic dictionary that you created with 
    # this function
    results_stats_dic = dict()
    
    # 1. Calculation of (Z) n_images - number of images
    results_stats_dic['n_images'] = len(results_dic)
    
    # 2. Calculation of (A) n_correct_dogs - number of correctly classified dog images
    results_stats_dic['n_correct_dogs'] = 0 # We could initialize all variables in 1 line
    for f in results_dic:
        if results_dic[f][3] == 1 and results_dic[f][4] == 1:
            results_stats_dic['n_correct_dogs'] += 1
    
    # 3. Calculation of (B) n_dogs_img - number of dog images
    results_stats_dic['n_dogs_img'] = 0
    for f in results_dic:
        if results_dic[f][3] == 1:
            results_stats_dic['n_dogs_img'] += 1

    # 4. Calculation of (C) n_correct_notdogs - number of correctly classified NON-dog images            
    results_stats_dic['n_correct_notdogs'] = 0  
    for f in results_dic:
        if results_dic[f][3] == 0 and results_dic[f][4] == 0:
            results_stats_dic['n_correct_notdogs'] += 1

    # 5. Calculation of (D) n_notdogs_img - number of NON-dog images
    results_stats_dic['n_notdogs_img'] = 0
    for f in results_dic:
        if results_dic[f][3] == 0:
            results_stats_dic['n_notdogs_img'] += 1

    # 6. Calculation of (E) n_correct_breed - number of correctly classified dog breeds                
    results_stats_dic['n_correct_breed'] = 0
    for f in results_dic:
        results_stats_dic['n_correct_breed'] += results_dic[f][2] * results_dic[f][3]

    # 7. Calculation of (Y) n_match - number of matches between pet & classifier labels        
    results_stats_dic['n_match'] = 0
    for f in results_dic:
        results_stats_dic['n_match'] += results_dic[f][2]

    # For percentages: watchout for the ZeroDivide!
    try:
        # 8. Calculation of (1a) pct_correct_dogs - percentage of correctly classified dogs = A/B*100
        if results_stats_dic['n_dogs_img'] != 0:
            results_stats_dic['pct_correct_dogs'] = 1.0 * results_stats_dic['n_correct_dogs'] * 100 / results_stats_dic['n_dogs_img']
        else:
            results_stats_dic['pct_correct_dogs'] = 0

        # 9. Calculation of (1b) pct_correct_notdogs - percentage of correctly classified NON-dogs = C/D*100
        if results_stats_dic['n_notdogs_img'] != 0:
            results_stats_dic['pct_correct_notdogs'] = 1.0 * results_stats_dic['n_correct_notdogs'] * 100 / results_stats_dic['n_notdogs_img']
        else:
            results_stats_dic['pct_correct_notdogs'] = 0

        #10. Calculation of (2) pct_correct_breed - percentage of correctly classified dog breeds = E/B*100
        if results_stats_dic['n_dogs_img'] != 0:
            results_stats_dic['pct_correct_breed'] = 1.0 * results_stats_dic['n_correct_breed'] * 100 / results_stats_dic['n_dogs_img']
        else:
            results_stats_dic['pct_correct_breed'] = 0

        #11. Calculation of opt pct_match - percentage of correct matches = Y/Z*100
        if results_stats_dic['n_images'] != 0:
            results_stats_dic['pct_match'] = 1.0 * results_stats_dic['n_match'] * 100 / results_stats_dic['n_images']
        else:
            results_stats_dic['pct_match'] = 0

    except Exception as e:
        print("An exception occured: {}".format(e))

    #print("Statistics generation completed!")
        
    return results_stats_dic
