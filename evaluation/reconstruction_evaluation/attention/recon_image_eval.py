'''Make figures of feature decoding results.'''


from glob import glob
import os
import itertools

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist

def recon_image_eval(
        recon_image_dir,
        true_image_dir,
        subjects=[],
        rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
        labels=None,
        attended_list=None,
        unattended_list=None
):
    print('Subjects:', subjects)
    print('ROIs:', rois)
    print('Reconstructed image dir:', recon_image_dir)
    print('True image dir:', true_image_dir)

    if not (labels and attended_list and unattended_list):
        raise ValueError("Must provide labels, attended_list, and unattended_list")

    # Get the reconstructed image size.
    sample_image_path = glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0]
    recon_image_size = Image.open(sample_image_path).size

    # Load trueA and trueB images in order.
    def read_image(img_index):
        path = os.path.join(true_image_dir, f"{img_index}.{true_image_ext}")
        return np.array(Image.open(path).resize(recon_image_size))

    trueA_list = [read_image(a) for a in attended_list]
    trueB_list = [read_image(b) for b in unattended_list]

    print(f"Loaded {len(trueA_list)} attended (trueA) images.")
    print(f"Loaded {len(trueB_list)} unattended (trueB) images.")

    for subject, roi in itertools.product(subjects, rois):
        print(f'Subject: {subject} - ROI: {roi}')
        current_recon_dir = os.path.join(recon_image_dir, subject, roi)

        pred_list = []
        for label in labels:
            recon_path = os.path.join(current_recon_dir, f'recon_image-{label}.0.{recon_image_ext}')
            matches = glob(recon_path)
            if len(matches) != 1:
                raise RuntimeError(f'Expected 1 recon image for label {label}, found {len(matches)}')
            pred_list.append(np.array(Image.open(matches[0])))

        acc, results = attention_identification(pred_list, trueA_list, trueB_list)
        print(f'Identification accuracy: {acc:.4f}')

    return acc, results


def attention_identification(pred_list, trueA_list, trueB_list, metric='correlation'):
    """
    Run binary identification for predicted images against trueA/trueB pairs.

    Parameters:
        pred_list: list of np.ndarray, predicted images or features.
        trueA_list: list of np.ndarray, correct option A.
        trueB_list: list of np.ndarray, distractor option B.
        metric: similarity metric, default is correlation.

    Returns:
        accuracy: overall identification accuracy.
        results: 0/1 list indicating whether each sample was identified correctly.
    """
    results = []
    for i in range(len(pred_list)):
        pred = pred_list[i].flatten().reshape(1, -1)
        trueA = trueA_list[i].flatten().reshape(1, -1)
        trueB = trueB_list[i].flatten().reshape(1, -1)

        sim_A = 1 - cdist(pred, trueA, metric=metric)[0, 0]
        sim_B = 1 - cdist(pred, trueB, metric=metric)[0, 0]

        is_correct = sim_A > sim_B
        results.append(1 if is_correct else 0)

    accuracy = np.mean(results)
    return accuracy, results


def get_recon_image_dir(src, trg, recon_base_dir):
    conversion = f'{src}_2_{trg}'
    return os.path.join(recon_base_dir, conversion)


def get_subject_pairs(subjects_list, example_pair=('sub02', 'sub01')):
    if example_pair is not None:
        return [example_pair]

    return itertools.permutations(subjects_list, 2)


def build_attention_image_lists(image_labels, exclude_labels):
    image_indices = [
        int(label)
        for label in image_labels
        if label not in exclude_labels
    ]

    labels = []
    attended_list = []
    unattended_list = []

    for idx in image_indices:
        attended = (idx - 1) // 10 + 1
        unattended = (idx - 1) % 10 + 1
        label = (attended - 1) * 10 + unattended

        labels.append(label)
        attended_list.append(attended)
        unattended_list.append(unattended)

    return labels, attended_list, unattended_list


def run_all_pairs(config):
    result_data = []

    labels, attended_list, unattended_list = build_attention_image_lists(
        config['image_labels'],
        config['exclude_labels']
    )

    subject_pairs = get_subject_pairs(
        config['subjects_list'],
        config['example_pair']
    )

    for src, trg in subject_pairs:
        recon_image_dir = get_recon_image_dir(
            src,
            trg,
            config['recon_base_dir']
        )

        mean_ident, results = recon_image_eval(
            recon_image_dir,
            config['true_image_dir'],
            subjects=config['subjects'],
            rois=config['rois'],
            labels=labels,
            attended_list=attended_list,
            unattended_list=unattended_list
        )

        result_data.append({
            'Source': src,
            'Target': trg,
            'Identification accuracy': mean_ident,
            'Method': config['method'],
            'ROI': config['rois'][0]
        })

    return result_data


def save_results(result_data, output_filename):
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    data_df = pd.DataFrame(result_data)
    data_df.to_csv(os.path.join(output_dir, output_filename), index=None)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    config = {
        'true_image_dir': os.path.join(project_root, 'data', 'test_image', 'attention'),
        'recon_base_dir': os.path.join(
            project_root,
            'reconstruction',
            'attention_imagery',
            'recon_attention'
        ),
        'subjects': ['target'],
        'rois': ['VC'],
        'subjects_list': ['sub01', 'sub02', 'sub03', 'sub04'],
        'image_labels': [str(i) for i in range(1, 101)],
        'exclude_labels': ['1', '12', '23', '34', '45', '56', '67', '78', '89', '100'],
        'method': 'Content_loss',
        'output_file': 'attention_image_pixel_identification.csv',
        # Set to None to run all subject pairs.
        'example_pair': ('sub02', 'sub01')
    }

    result_data = run_all_pairs(config)
    save_results(result_data, config['output_file'])

    print('All done')


if __name__ == '__main__':
    main()
