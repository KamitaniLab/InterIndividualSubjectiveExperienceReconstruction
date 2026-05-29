import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import bdpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
ATTENTION_NCC_DIR = os.path.join(PROJECT_ROOT, 'neural_code_conversion', 'attention')

# Add the attention neural code conversion directory for module imports.
sys.path.insert(0, ATTENTION_NCC_DIR)
from models import Converter
from utils import PathBuilder, fastl2lir_parameter, load_mean_std


def load_brain_data(brain_dir, subjects_list):
    """
    Load brain data for each subject from the specified directory.
    """
    return {subject: bdpy.BData(os.path.join(brain_dir, dat_file)) for subject, dat_file in subjects_list.items()}


def compute_roi_index(data, embeded_roi, roi_mapping):
    """
    Compute the ROI indices to aggregate conversion matrices for each brain area.
    """
    _, base_idx = data.select(embeded_roi, return_index=True)
    idx_loc = np.where(base_idx)[0]
    idx_mapper = dict(zip(idx_loc, range(len(idx_loc))))

    idxs = {}
    for roi, roi_str in roi_mapping.items():
        _, idx = data.select(roi_str, return_index=True)
        loc = np.where(idx)[0]
        idxs[roi] = [idx_mapper[l] for l in loc]

    return idxs


def initialize_model(input_nc, output_nc, model_path):
    """
    Initialize and load the conversion model.
    """
    netG_A2B = Converter(input_nc, output_nc)
    netG_A2B.cuda()
    netG_A2B.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    return netG_A2B


def get_valid_trials(data, roi_str, label_name='Label'):
    """
    Extract valid trials for a given subject and ROI.

    Parameters:
        data_brain : dict
            Dictionary of subject data, e.g., data_brain['sub01'].
        subject : str
            Subject ID, e.g., 'sub01'.
        rois_list : dict
            ROI selection dictionary, e.g., {'VC': 'ROI_VC = 1'}.
        roi_key : str
            The key to select which ROI to use (default: 'VC').
        label_name : str
            Name of the label field to extract (default: 'Label').

    Returns:
        x : np.ndarray
            Filtered voxel data for the selected ROI.
        x_labels : np.ndarray
            Labels corresponding to the valid trials (only column 8).
        response : np.ndarray
            Response data (button presses).
        valid_index : np.ndarray
            Boolean index array indicating valid trials (attention trials only).
    """
    # data = data_brain[subject]

    # Extract original voxel data and labels
    x_original = data.select(roi_str)
    x_labels_original = data.select(label_name)

    # Extract trial information
    att_target = data.select("attention_target").flatten()
    response = data.select("response").flatten()

    # Valid trial index: attention trials only (exclude image trials where att_target == 0)
    valid_index = (att_target != 0)

    # Apply the valid trial mask
    x = x_original[valid_index]
    x_labels = x_labels_original[valid_index, 8]  # keep only the 9th column (index 8)

    return x, x_labels, response[valid_index], valid_index


def sort_and_convert_labels(x, x_labels):
    """
    Sorts datasets `x` and `y` based on their corresponding labels and converts labels to numeric.

    Parameters:
        x (np.ndarray): Dataset 1, shape `(n_samples_x, n_features)`.
        y (np.ndarray): Dataset 2, shape `(n_samples_y, n_features)`.
        x_labels (np.ndarray): Labels for dataset `x`, shape `(n_samples_x,)` or `(n_samples_x, 1)`.
        y_labels (np.ndarray): Labels for dataset `y`, shape `(n_samples_y,)` or `(n_samples_y, 1)`.

    Returns:
        x (np.ndarray): Sorted dataset `x`.
        y (np.ndarray): Sorted dataset `y`.
        numeric_label_x (np.ndarray): Sorted numeric labels corresponding to `x`.
        numeric_label_y (np.ndarray): Sorted numeric labels corresponding to `y`.
    """
    x_labels = np.asarray(x_labels).flatten()

    # # Get unique labels and map them to numeric values
    # unique_labels = np.unique(x_labels)  # Get unique labels.
    # label_mapping = {lb: idx + 1 for idx, lb in enumerate(unique_labels)} # Build the label mapping.
    # numeric_label_x = np.array([label_mapping[lb] for lb in x_labels])

    # Sort x and numeric_label_x
    x_index = np.argsort(x_labels.flatten())
    numeric_label_x = x_labels[x_index]
    x = x[x_index, :]

    return x, numeric_label_x


def process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, rep, num_samples=6000):
    """
    Process a pair of subjects, perform conversion, and calculate correlation.
    """
    dat1 = data_brain[src]
    dat2 = data_brain[trg]

    x_original_valid, x_labels_valid, _, _ = get_valid_trials(dat1, roi_dict[base_ROI])
    y_original_valid, y_labels_valid, _, _ = get_valid_trials(dat2, roi_dict[base_ROI])

    input_nc, output_nc = x_original_valid.shape[1], y_original_valid.shape[1]

    conversion = f'{src}_2_{trg}'
    print(f'Source: {src}, Target: {trg}')

    # Initialize and load the conversion model.
    # For a locally trained model, use:
    # converter_roi_dir = os.path.join(PROJECT_ROOT, 'neural_code_conversion', 'attention', 'output', conversion, base_ROI)
    converter_roi_dir = os.path.join(PROJECT_ROOT, 'data', 'pre-trained', 'converters', 'attention_imagery', conversion, base_ROI)
    model_path = os.path.join(converter_roi_dir, 'model.pth')
    netG_A2B = initialize_model(input_nc, output_nc, model_path)

    # Align samples and sort them.
    x, x_test_labels = sort_and_convert_labels(x_original_valid, x_labels_valid)
    y, y_test_labels = sort_and_convert_labels(y_original_valid, y_labels_valid)

    # Load source normalization from the converter and target normalization from the target decoder.
    path_trg = PathBuilder(pre_vgg_models_dir_root, vgg_network, trg, base_ROI)
    x_mean_src, x_norm_src = load_mean_std(os.path.join(converter_roi_dir, 'norm'))
    x_mean_trg, x_norm_trg = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)[:2]

    x_item = (x - x_mean_src) / x_norm_src
    real_A = Variable(torch.cuda.FloatTensor(x_item), requires_grad=False)

    # Generate output
    fake_B = netG_A2B(real_A).detach().cpu().numpy()
    converted_x = x_norm_trg * fake_B + x_mean_trg

    # Compute ROI indices and calculate correlations
    y_roi_idxs = compute_roi_index(dat2, roi_dict[base_ROI], roi_dict)
    calculate_correlations(y, converted_x, y_roi_idxs, src, trg, result_data, rep, num_samples)


def calculate_correlations(y, converted_x, y_roi_idxs, src, trg, result_data, rep, num_samples):
    """
    Calculate correlations for each voxel and store the results.
    """
    for trg_roi, roi_idxs in y_roi_idxs.items():
        y_sub = y[:, roi_idxs]
        converted_x_sub = converted_x[:, roi_idxs]

        for i in range(y_sub.shape[1]):
            y_sub_vox = y_sub[:, i].reshape(rep, -1, order='F')
            converted_x_sub_vox = converted_x_sub[:, i].reshape(rep, -1, order='F')
            corr = np.mean(np.corrcoef(y_sub_vox, converted_x_sub_vox)[rep:, :rep])

            result_data.append({
                'Source': src,
                'Target': trg,
                'Number of samples': num_samples,
                'Correlation': corr,
                'Method': 'content_loss',
                'ROI': trg_roi,
                'Vox_idx': i,
                'Target ROI': trg_roi
            })


def save_results(result_data, output_dir, output_filename):
    """
    Save the results to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(result_data)
    df.to_csv(os.path.join(output_dir, output_filename), index=None)
    print(f'Results saved to {os.path.join(output_dir, output_filename)}')


def get_subject_pairs(subjects_list, example_pair=None):
    """
    Return the default example pair or all source-target subject pairs.
    """
    if example_pair is not None:
        return [example_pair]

    return itertools.permutations(subjects_list.keys(), 2)


def main():
    """
    Main function to execute the process and save the results.
    """
    # Constants and paths
    brain_dir = os.path.join(PROJECT_ROOT, 'data', 'fmri', 'attention')
    vgg_network = 'caffe/VGG_ILSVRC_19_layers'
    pre_vgg_models_dir_root = os.path.join(
        PROJECT_ROOT,
        'data',
        'pre-trained',
        'decoders',
        'attention_imagery',
        'deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100'
    )
    output_dir = './results'
    output_filename = 'attention_conversion_accuracy_profile.csv'

    # Constants
    base_ROI = 'VC'
    trials_per_img = 8
    example_pair = ('sub02', 'sub01')  # Set to None to evaluate all subject pairs.

    # Subjects list
    subjects_list = {'sub01': 'sub-01_attention.h5',
                     'sub02': 'sub-02_attention.h5',
                     'sub03': 'sub-03_attention.h5',
                     'sub04': 'sub-04_attention.h5',
                     }

    # ROI dictionary
    roi_dict = {
        'VC': 'ROI_VC =1',
        'V1': 'ROI_V1 = 1',
        'V2': 'ROI_V2 = 1',
        'V3': 'ROI_V3 = 1',
        'V4': 'ROI_hV4 = 1',
        'HVC': 'ROI_HVC = 1'
    }

    data_brain = load_brain_data(brain_dir, subjects_list)
    result_data = []

    for src, trg in get_subject_pairs(subjects_list, example_pair):
        process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, trials_per_img)

    save_results(result_data, output_dir, output_filename)


if __name__ == "__main__":
    main()
