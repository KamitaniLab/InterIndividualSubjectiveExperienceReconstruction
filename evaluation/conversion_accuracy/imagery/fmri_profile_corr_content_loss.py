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
IMAGERY_NCC_DIR = os.path.join(PROJECT_ROOT, 'neural_code_conversion', 'imagery')

# Add the imagery neural code conversion directory for module imports.
sys.path.insert(0, IMAGERY_NCC_DIR)
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


def sort_and_convert_labels(x, x_labels, rep=19):
    """
    Sort the dataset `x` according to its labels, convert labels to numeric indices,
    and ensure each label keeps only the first shared `rep` trials.

    Parameters
    ----------
    x : np.ndarray
        Data matrix of shape (n_samples, n_features).
    x_labels : np.ndarray
        Label vector of shape (n_samples,) or (n_samples, 1).
    rep : int
        Number of trials to keep per label (default: 19).

    Returns
    -------
    x_out : np.ndarray
        Filtered and sorted data matrix where each label has exactly `rep` trials.
    labels_out : np.ndarray
        Corresponding numeric labels after sorting and filtering.
    """

    # --- 1. Convert labels to 1D array ---
    x_labels = np.asarray(x_labels).flatten()

    # --- 2. Remove blank trials (label == 16) ---
    # This ensures only valid image trials are kept.
    mask = x_labels != 16.0
    x = x[mask]
    x_labels = x_labels[mask]

    # --- 3. Get unique labels and map them to numeric indices ---
    # Example: {101:1, 102:2, 105:3, ...}
    unique_labels = np.unique(x_labels)
    label_mapping = {lb: idx + 1 for idx, lb in enumerate(unique_labels)}

    # Convert original labels into numeric labels
    numeric_x_labels = np.array([label_mapping[lb] for lb in x_labels])

    # --- 4. Sort x and labels according to the numeric labels ---
    sort_index = np.argsort(numeric_x_labels)
    x = x[sort_index, :]
    numeric_x_labels = numeric_x_labels[sort_index]

    # --- 5. For each label, keep only the first `rep` trials ---
    final_x = []
    final_labels = []

    for lb in np.unique(numeric_x_labels):

        # Find indices where label == lb
        label_mask = numeric_x_labels == lb
        idx = np.where(label_mask)[0]

        # If label has fewer than `rep` trials, skip it or raise warning
        if len(idx) < rep:
            print(f"Warning: label {lb} has only {len(idx)} trials (<{rep}), skipping.")
            continue

        # Keep the first `rep` trials for this label
        keep_idx = idx[:rep]
        final_x.append(x[keep_idx])
        final_labels.append(numeric_x_labels[keep_idx])

    # --- 6. Stack results back into final array ---
    x_out = np.vstack(final_x)
    labels_out = np.concatenate(final_labels)

    return x_out, labels_out


def process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, rep, num_samples=6000):
    """
    Process a pair of subjects, perform conversion, and calculate correlation.
    """
    dat1 = data_brain[src]
    dat2 = data_brain[trg]

    x_test, _ = dat1.select(roi_dict[base_ROI], return_index=True)
    y_test, _ = dat2.select(roi_dict[base_ROI], return_index=True)
    input_nc, output_nc = x_test.shape[1], y_test.shape[1]

    conversion = f'{src}_2_{trg}'
    print(f'Source: {src}, Target: {trg}')

    # Initialize and load the conversion model.
    # For a locally trained model, use:
    # converter_roi_dir = os.path.join(PROJECT_ROOT, 'neural_code_conversion', 'imagery', 'output', conversion, base_ROI)
    converter_roi_dir = os.path.join(PROJECT_ROOT, 'data', 'pre-trained', 'converters', 'attention_imagery', conversion, base_ROI)
    model_path = os.path.join(converter_roi_dir, 'model.pth')
    netG_A2B = initialize_model(input_nc, output_nc, model_path)

    # Extract test labels and unique labels
    x_labels = dat1.select('imagery_image_index')
    y_labels = dat2.select('imagery_image_index')

    # Align samples and sort them.
    x, x_test_labels = sort_and_convert_labels(x_test, x_labels)
    y, y_test_labels = sort_and_convert_labels(y_test, y_labels)

    # Combine data by unique labels
    # x = np.vstack([x_test[(x_test_labels == lb).flatten(), :] for lb in x_test_labels_unique])
    # y = np.vstack([y_test[(y_test_labels == lb).flatten(), :] for lb in y_test_labels_unique])

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
    brain_dir = os.path.join(PROJECT_ROOT, 'data', 'fmri', 'imagery')
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
    output_filename = 'imagery_conversion_accuracy_profile.csv'

    # Constants
    base_ROI = 'VC'
    trials_per_img = 19
    example_pair = ('sub02', 'sub01')  # Set to None to evaluate all subject pairs.

    # Subjects list
    subjects_list = {'sub01': 'sub-01_imagery.h5',
                     'sub02': 'sub-02_imagery.h5',
                     'sub03': 'sub-03_imagery.h5',
                     'sub04': 'sub-04_imagery.h5',
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
