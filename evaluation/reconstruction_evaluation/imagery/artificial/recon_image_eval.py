'''Make figures of feature decoding results.'''


import argparse
from glob import glob
from itertools import product
import os,itertools

from bdpy.evals.metrics import pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import pandas as pd
from PIL import Image
import yaml


def recon_image_eval(
        recon_image_dir,
        true_image_dir,
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='tiff',
        image_labels=None  # Optional image label list without extensions.
):
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Reconstructed image dir: {}'.format(recon_image_dir))
    print('True images dir:         {}'.format(true_image_dir))
    print('')

    # Use any reconstructed image to determine the image size.
    img = Image.open(glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0])
    recon_image_size = img.size

    if image_labels is not None:
        true_image_files = [
            os.path.join(true_image_dir, f"{label}.{true_image_ext}")
            for label in image_labels
        ]
        true_image_labels = image_labels
    else:
        all_true_image_files = glob(os.path.join(true_image_dir, '*.' + true_image_ext))
        true_image_files = sorted(all_true_image_files)
        true_image_labels = [
            os.path.splitext(os.path.basename(f))[0]
            for f in true_image_files
        ]
    true_images = np.vstack([
        np.array(Image.open(f).resize(recon_image_size)).flatten()
        for f in true_image_files
    ])
    for subject, roi in itertools.product(subjects, rois):
        print(f'Subject: {subject} - ROI: {roi}')
        current_recon_dir = os.path.join(recon_image_dir, subject, roi)

        recon_images = []
        for label in true_image_labels:
            pattern = os.path.join(current_recon_dir, f'recon_image-{label}.{recon_image_ext}')
            matches = glob(pattern)

            if len(matches) == 0:
                raise RuntimeError(f'No reconstructed image found for label: {label}')
            elif len(matches) > 1:
                raise RuntimeError(f'Multiple reconstructed images found for label: {label}: {matches}')

            recon_img = Image.open(matches[0])
            recon_images.append(np.array(recon_img).flatten())

        recon_images = np.vstack(recon_images)

        # Compute evaluation metrics.
        r_pixelt = pattern_correlation(recon_images, true_images)
        ident = pairwise_identification(recon_images, true_images)

        print('Mean pixel correlation:       {:.4f}'.format(np.nanmean(r_pixelt)))
        print('Mean identification accuracy: {:.4f}'.format(np.nanmean(ident)))

    return np.nanmean(r_pixelt), np.nanmean(ident)


def get_recon_image_dir(src, trg, recon_base_dir):
    conversion = f'{src}_2_{trg}'
    return os.path.join(recon_base_dir, conversion)


def get_subject_pairs(subjects_list, example_pair=('sub02', 'sub01')):
    if example_pair is not None:
        return [example_pair]

    return itertools.permutations(subjects_list, 2)


def run_all_pairs(config):
    result_data = []

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

        r_pixel, ident = recon_image_eval(
            recon_image_dir,
            config['true_image_dir'],
            subjects=config['subjects'],
            rois=config['rois'],
            image_labels=config['image_labels']
        )

        result_data.append({
            'Source': src,
            'Target': trg,
            'Identification accuracy': ident,
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))

    config = {
        'true_image_dir': os.path.join(project_root, 'data', 'test_image', 'imagery'),
        'recon_base_dir': os.path.join(
            project_root,
            'reconstruction',
            'attention_imagery',
            'recon_imagery'
        ),
        'subjects': ['target'],
        'rois': ['VC'],
        'subjects_list': ['sub01', 'sub02', 'sub03', 'sub04'],
        'image_labels': [
            '1.0', '2.0', '3.0', '4.0', '5.0',
            '6.0', '7.0', '8.0', '9.0', '10.0',
            '11.0', '12.0', '13.0', '14.0', '15.0'
        ],
        'method': 'Content_loss',
        'output_file': 'artificial_image_pixel_identification.csv',
        # Set to None to run all subject pairs.
        'example_pair': ('sub02', 'sub01')
    }

    result_data = run_all_pairs(config)
    save_results(result_data, config['output_file'])

    print('All done')


if __name__ == '__main__':
    main()
