#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:10:25 2022

@author: fcheng
"""


import glob
import os

import numpy as np
import pandas as pd
import itertools

from eval.image_process import img_process
from eval.make_regressor import MakeRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# save dirs
save_dir = "./results/evaluation"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    
image_keys = ['Ehrenstein', 'Varin']  
# Target images                                                                                                                                                                         
images_set = dict.fromkeys(image_keys)
images_set['Ehrenstein'] = [
           'fillingin001_small_unionjack_lumi0p3_sat0p8_connected',
           'fillingin002_small_unionjack_lumi0p3_sat0p8_disconnected',
           'fillingin004_large_unionjack_lumi0p3_sat0p8_connected',
           'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected',
           'fillingin007_small_cross_lumi0p3_sat0p7_connected',
           'fillingin008_small_cross_lumi0p3_sat0p7_disconnected',
           'fillingin010_large_cross_lumi0p3_sat0p7_connected',
           'fillingin011_large_cross_lumi0p3_sat0p7_disconnected'
           ]
           
images_set['Varin'] =  [
                        'neonVarinImg001_IllusSurf_alpha_0p7_sat_0p3_bgray',     
                        'neonVarinImg002_innerKanizsa_alpha_0p7_sat_0p3_bgray',    
                        ]
# stimulus type
stimtypes = ['Illusion', 'Control']
stimType_list = dict.fromkeys(image_keys)
stimType_list['Ehrenstein'] = ['Illusion', 'Control', 'Illusion', 'Control', 
                               'Illusion', 'Control', 'Illusion', 'Control']
stimType_list['Varin'] = ['Illusion', 'Control']

# pattern type
pattern_list = dict.fromkeys(image_keys)
pattern_list['Ehrenstein'] = ['unionjack', 'unionjack', 'unionjack', 'unionjack',
                              'cross', 'cross', 'cross', 'cross']
pattern_list['Varin'] = ['Illusory surface','Inner Kanizsa']

# size 
size_list = dict.fromkeys(image_keys)
size_list['Ehrenstein'] = [3, 3, 9, 9, 3, 3, 9, 9]
size_list['Varin'] = [6,6] 

# parameters
img_size = 227
n = img_size**2
normalization = None

# presented images
stimuli_dir_root = os.path.join(PROJECT_ROOT, 'data', 'test_image', 'illusion', 'source')
source_image_dir = stimuli_dir_root
source_image_ext = "tif"


path = os.path.join(
    PROJECT_ROOT,
    'reconstruction',
    'illusion',
    'results',
    'reconstruction',
    'recon_images_single_trial',
    'GAN'
)

def get_subject_pairs(subjects_list, example_pair=('sub02', 'sub01')):
    if example_pair is not None:
        return [example_pair]

    return itertools.permutations(subjects_list, 2)


# recon from decoded features
sbjs = []
subjects_list = ['sub01','sub02','sub03','sub04']
# Default example: sub02_2_sub01.
# Set to None to run all subject pairs.
example_pair = ('sub02', 'sub01')
for src, trg in get_subject_pairs(subjects_list, example_pair):
    conversion = f"{src}_2_{trg}"
    sbjs.append(conversion)
rois = ['VC']

# regressor 
path_to_regressor = os.path.join(PROJECT_ROOT, 'data', 'test_image', 'illusion', 'regressor')
maptype = 'Redness'
label = 'stimulus + red surface'
regressor = ['stimulus', 'red_surface']

# Main #######################################################################
for figtype in image_keys:
    
    # images information
    images = images_set[figtype]
    stimtype = stimType_list[figtype]

    # Initialization fot dataframe
    Weight = []
    stimType = []
    Model = []
    Map = []
    ROI = []
    Subject = []
    reconType = []
    Trial = []
    stimName = []

    # recon from decoded features --------------------------------------
    for sbj in sbjs:

        print('Subject:{0}'.format(sbj))
        
        if figtype == 'Varin' and sbj.startswith('TH_2_'):
            continue
     
        for roi in rois:
            print('ROI:{0}'.format(roi))
            

            # prepare response vector
            n_sample = n 
            # prepare predictor matrix 
            k = len(regressor)
            

            for i, image in enumerate(images):

                # make predictor matrix for each stimulus type  
                X = MakeRegressor(image, path_to_regressor, regressor, img_size, 
                                  normalization=normalization, redness=maptype, interception=False)
                # prepare response vector
                img_path = os.path.join(path, sbj, 'target', roi, 'recon_image_normalized-'+image + "*.tiff")
                imgfiles = sorted(glob.glob(img_path))
                for f, fn in enumerate(imgfiles): 
                    x = np.zeros((n_sample, k+1))
                    y = np.zeros((n_sample, 1))
                    Y = img_process(fn, img_size, redness=maptype)     

                    y[:, 0] = Y.flatten()/255
                    x = X    
             
                    # linear regression
                    w = np.linalg.lstsq(x , y, rcond=None)[0]
                    Weight.append(w)
                    print("w:{}".format(w))
                    
 
                    # model parameter
                    Model.append(label)
                    stimType.append(stimtype[i])
                    stimName.append(image)
                                   
                    # fmri parameter
                    reconType.append('Recon-decoded features')
                    ROI.append(roi)
                    Subject.append(sbj)
                    Trial.append(f+1)
                    
        
    # save results -----------------------------------------------------------
    save_title = 'Regression_color_'+ figtype
    reg_pkl_file = os.path.join(save_dir, save_title+'.pkl')  
    reg = pd.DataFrame.from_dict({ 
                                    'Beta coefficient': Weight, 
                                    'Trial':Trial, 'stimName': stimName, 'stimType':stimType,'reconType':reconType,
                                   'Model':Model,'ROI': ROI, 'Subject': Subject, 
                                  })
    reg.to_pickle(reg_pkl_file)
               
              
    
print('All done')












