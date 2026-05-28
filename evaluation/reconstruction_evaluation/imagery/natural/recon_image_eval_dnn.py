import os
import itertools
from glob import glob
from PIL import Image
import copy

import numpy as np
import pandas as pd
import torch

from bdpy.evals.metrics import pairwise_identification
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import VGG19, AlexNet, layer_map

# Main evaluation function ####################################################

def recon_image_eval_dnn(
        recon_image_dir,
        true_image_dir,
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='tiff',
        recon_eval_encoder='AlexNet',
        device='cuda:0',
        image_labels=None  # Supports image filtering.
):
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('Reconstructed image dir: {}'.format(recon_image_dir))
    print('True images dir:         {}'.format(true_image_dir))
    print('Evaluation encoder: {}'.format(recon_eval_encoder))
    print('')

    # Get the reconstructed image size.
    ref_image = glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0]
    img = Image.open(ref_image)
    recon_image_size = img.size

    # === Load true images ===
    if image_labels is not None:
        true_image_files = [
            os.path.join(true_image_dir, f"{label}.{true_image_ext}")
            for label in image_labels
        ]
        true_image_labels = image_labels
    else:
        true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
        true_image_labels = [
            os.path.splitext(os.path.basename(f))[0]
            for f in true_image_files
        ]

    true_images = [
        Image.open(f).convert("RGB").resize(recon_image_size, Image.LANCZOS)
        for f in true_image_files
    ]

    # === Load the DNN model ===
    dnnh = DNNHandler(recon_eval_encoder, device=device)

    for subject, roi in itertools.product(subjects, rois):
        print(f'DNN: {recon_eval_encoder} - Subject: {subject} - ROI: {roi}')
        current_recon_dir = os.path.join(recon_image_dir, subject, roi)

        recon_images = []
        for label in true_image_labels:
            recon_path = os.path.join(current_recon_dir, f'recon_image-{label}.{recon_image_ext}')
            if not os.path.exists(recon_path):
                raise RuntimeError(f'Reconstructed image not found for label: {label} at {recon_path}')
            recon_img = Image.open(recon_path).convert("RGB")
            recon_images.append(recon_img)

        # === Compute DNN features ===
        true_feat = dnnh.get_activation(true_images, flat=True)
        recon_feat = dnnh.get_activation(recon_images, flat=True)

    return true_feat, recon_feat, dnnh.layers


# DNN wrapper class ###########################################################

class DNNHandler():
    def __init__(self, encoder_name="AlexNet", device='cpu'):
        self.encoder_name = encoder_name
        self.device = device

        if encoder_name == "AlexNet":
            self.encoder = AlexNet()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt'
        elif encoder_name == "VGG19":
            self.encoder = VGG19()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
        else:
            raise RuntimeError("Unsupported DNN model:", encoder_name)

        self.encoder.to(device)
        self.encoder.load_state_dict(torch.load(encoder_param_file))
        self.encoder.eval()

        self.image_size = [227, 227] if encoder_name == "AlexNet" else [224, 224]
        self.mean_image = np.float32([104., 117., 123.])
        self.layer_mapping = layer_map(encoder_name.lower())
        self.layers = list(self.layer_mapping.keys())
        self.feature_extractor = FeatureExtractor(self.encoder, self.layers, self.layer_mapping, device=self.device, detach=True)

    def get_activation(self, img_obj_list, flat=False):
        _img_obj_list = copy.deepcopy(img_obj_list)
        if not isinstance(_img_obj_list, list):
            _img_obj_list = [_img_obj_list]
        if isinstance(_img_obj_list[0], np.ndarray):
            _img_obj_list = [Image.fromarray(a_img) for a_img in _img_obj_list]

        activations = {layer: [] for layer in self.layers}
        for a_img in _img_obj_list:
            a_img = a_img.resize(self.image_size, Image.LANCZOS)
            x = np.asarray(a_img)
            x = np.transpose(x, (2, 0, 1))[::-1]
            x = np.float32(x) - np.reshape(self.mean_image, (3, 1, 1))
            features = self.feature_extractor.run(x)
            for layer in self.layers:
                activations[layer].append(features[layer])

        for layer in self.layers:
            activations[layer] = np.vstack(activations[layer])
            if flat:
                activations[layer] = activations[layer].reshape(activations[layer].shape[0], -1)

        return activations


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

        try:
            true_feat, recon_feat, layers = recon_image_eval_dnn(
                recon_image_dir,
                config['true_image_dir'],
                subjects=config['subjects'],
                rois=config['rois'],
                image_labels=config['image_labels'],
                recon_eval_encoder=config['recon_eval_encoder']
            )
        except Exception as e:
            print(f"[ERROR] {src} -> {trg}: {e}")
            continue

        for layer in layers:
            ident = pairwise_identification(recon_feat[layer], true_feat[layer])
            print(f"Layer: {layer} | Mean ID Accuracy: {np.nanmean(ident):.4f}")

            result_data.append({
                'Source': src,
                'Target': trg,
                'Identification accuracy': np.nanmean(ident),
                'Method': config['method'],
                'ROI': config['rois'][0],
                'Layer': layer
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
        'recon_eval_encoder': 'AlexNet',
        'image_labels': [
            '17.0', '18.0', '19.0', '20.0', '21.0',
            '22.0', '23.0', '24.0', '25.0', '26.0'
        ],
        'method': 'Content_loss',
        'output_file': 'natural_image_dnn_identification.csv',
        # Set to None to run all subject pairs.
        'example_pair': ('sub02', 'sub01')
    }

    result_data = run_all_pairs(config)
    save_results(result_data, config['output_file'])

    print('All done')


if __name__ == '__main__':
    main()
