import argparse
import os
from torch.autograd import Variable
import torch
from time import time
from models import Converter
import numpy as np
from bdpy.dataform import save_array
from bdpy.util import makedir_ifnot
from utils import fastl2lir_parameter, test_fastl2lir_revise, load_mean_std
import itertools, bdpy
from utils import PathBuilder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def parse_arguments():
    """
    Parse command-line arguments and return an object containing them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    return parser.parse_args()

def load_data(brain_dir, subjects_list):
    """
    Load brain data from the specified directory.
    """
    return {subject: bdpy.BData(os.path.join(brain_dir, dat_file)) for subject, dat_file in subjects_list.items()}

def setup_environment(opt):
    """
    Set environment variables and check for CUDA availability.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def convert_brain_activity(subject_src, subject_trg, roi, data_brain, rois_list, trg_decoder_dir, vgg_network, opt):
    """
    Convert brain activity data and save the results.

    Parameters:
    - subject_src: The source subject ID.
    - subject_trg: The target subject ID.
    - roi: The name of the region of interest (ROI).
    - data_brain: A dictionary containing each subject's brain data.
    - rois_list: A dictionary containing ROI names and selection criteria.
    - trg_decoder_dir: Target decoder directory.
    - vgg_network: The type of VGG network used.
    - opt: An object containing the command-line arguments.
    """
    conversion = f"{subject_src}_2_{subject_trg}"
    print(f'Source: {subject_src}')
    print(f'Target: {subject_trg}')
    print(f'Conversion: {conversion}')
    print(f'ROI: {roi}')

    # Select ROI data and labels from the source subject
    x = data_brain[subject_src].select(rois_list[roi])
    x_labels =  data_brain[subject_src].select('imagery_image_index')

    # Build paths
    path_trg = PathBuilder(trg_decoder_dir, vgg_network, subject_trg, roi)

    # Get normalization parameters for the source and target
    x_mean_trg, x_norm_trg, _, _, _, _ = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)

    # The output dimension matches the target's normalization parameters
    input_nc = x.shape[1]
    output_nc = x_mean_trg.shape[1]
    netG_A2B = Converter(input_nc, output_nc)

    # If using GPU, move the model to the GPU
    if opt.cuda:
        netG_A2B.cuda()

    # Load the pretrained model's weights
    # If you use the model trained from scratch by yourself, the directory should be 
    # converter_dir = os.path.join('output', conversion)
    converter_dir = os.path.join(PROJECT_ROOT, 'data/pre-trained/converters/attention_imagery', conversion)
    device = torch.device('cuda:0')
    netG_A2B.load_state_dict(torch.load(os.path.join(converter_dir, roi, 'model.pth'), map_location=device))
    netG_A2B.eval()
    x_mean_src, x_norm_src = load_mean_std(os.path.join(converter_dir, roi, 'norm'))

    # Define Tensor type
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

    # Get unique labels and compute their averages
    x_test_labels_unique = np.unique(x_labels)
    x_averaged = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0) for lb in x_test_labels_unique])
    x_item = (x_averaged - x_mean_src) / x_norm_src
    real_A = Variable(Tensor(x_item), requires_grad=False)

    # Convert brain activity data
    fake_B = netG_A2B(real_A)
    pred_B = fake_B.detach().cpu().numpy()
    y_pred = x_norm_trg * pred_B + x_mean_trg

    # Define the list of VGG features to decode
    features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                     'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                     'fc6', 'fc7', 'fc8'][::-1]

    # Decode features and save results
    for vgg_feat in features_list:
        start_time = time()
        pred_dnn = test_fastl2lir_revise(path_trg.build_model_path(vgg_feat), path_trg.build_model_path(vgg_feat), y_pred)
        print(f'Total elapsed time (prediction): {time() - start_time:.6f} seconds')
        print(f'VGG feature: {vgg_feat}')

        results_dir_root = './result_vgg'
        results_dir_prediction = os.path.join(results_dir_root, conversion, vgg_network, vgg_feat, "target", roi)
        makedir_ifnot(results_dir_prediction)

        for i, label in enumerate(x_test_labels_unique):
            feature = np.array([pred_dnn[i,]])
            save_file = os.path.join(results_dir_prediction, f'{label}.mat')
            save_array(save_file, feature, 'feat', dtype=np.float32, sparse=False)

        print(f'Saved: {results_dir_prediction}')

def main():
    """
    Main function to execute the program logic by calling other functions.
    """
    opt = parse_arguments()
    setup_environment(opt)

    # Set the brain data path and subjects list
    brain_dir = os.path.join(PROJECT_ROOT, 'data/fmri/imagery')
    # brain_dir = '/home/kiss/data/fmri_shared/datasets/Attention/fmriprep'
    subjects_list = {'sub01': 'TH_Imagery_volume_native.h5',
                     'sub02': 'AM_Imagery_volume_native.h5',
                     # 'sub03': 'ES_Imagery_volume_native.h5',
                     # 'sub04': 'KS_Imagery_volume_native.h5',
                     }
    
    data_brain = load_data(brain_dir, subjects_list)

    # Set VGG network and decoder paths
    vgg_network = 'caffe/VGG_ILSVRC_19_layers'
    trg_decoder_dir = os.path.join(PROJECT_ROOT,'data/pre-trained/decoders/attention_imagery/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100')
    
    # Define the list of regions of interest (ROI)
    rois_list = {'VC': 'ROI_VC =1'}

    # Convert brain activity data for each subject combination and ROI
    # for src, trg in itertools.permutations(subjects_list.keys(), 2):
    src, trg = 'sub02', 'sub01'
    for roi in rois_list:
        convert_brain_activity(src, trg, roi, data_brain, rois_list, trg_decoder_dir, vgg_network, opt)

if __name__ == "__main__":
    main()
