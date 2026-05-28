import os
import itertools
from glob import glob
from PIL import Image
import copy
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import VGG19, AlexNet, layer_map


# DNN feature extraction handler #############################################
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
        self.encoder.load_state_dict(torch.load(encoder_param_file, map_location=device))
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


# Binary attention identification ############################################
def attention_identification(pred_list, trueA_list, trueB_list, metric='correlation'):
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


# Evaluation function #########################################################
def recon_image_eval_dnn_attention(
    recon_image_dir, true_image_dir,
    subjects, rois,
    labels, attended_list, unattended_list,
    recon_image_ext='tiff', true_image_ext='JPEG',
    recon_eval_encoder='AlexNet', device='cpu'
):
    dnnh = DNNHandler(recon_eval_encoder, device=device)

    sample_image_path = glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0]
    recon_image_size = Image.open(sample_image_path).size

    def read_image(idx):
        path = os.path.join(true_image_dir, f"{idx}.{true_image_ext}")
        return Image.open(path).convert("RGB").resize(recon_image_size, Image.LANCZOS)

    trueA_imgs = [read_image(a) for a in attended_list]
    trueB_imgs = [read_image(b) for b in unattended_list]
    print(f"[INFO] Loaded {len(trueA_imgs)} attended and {len(trueB_imgs)} unattended images.")

    results_all = []

    for subject, roi in itertools.product(subjects, rois):
        pred_imgs = []
        for label in labels:
            path = os.path.join(recon_image_dir, subject, roi, f'recon_image-{label}.0.{recon_image_ext}')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            pred_imgs.append(Image.open(path).convert("RGB"))

        true_feat_A = dnnh.get_activation(trueA_imgs, flat=True)
        true_feat_B = dnnh.get_activation(trueB_imgs, flat=True)
        pred_feat = dnnh.get_activation(pred_imgs, flat=True)

        results = []
        for layer in dnnh.layers:
            acc, _ = attention_identification(
                pred_feat[layer], true_feat_A[layer], true_feat_B[layer]
            )
            results.append((layer, acc))
        results_all.append((subject, roi, results))

    return results_all


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

    labels, attended_list, unattended_list = [], [], []
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

        try:
            layer_results = recon_image_eval_dnn_attention(
                recon_image_dir,
                config['true_image_dir'],
                subjects=config['subjects'],
                rois=config['rois'],
                labels=labels,
                attended_list=attended_list,
                unattended_list=unattended_list,
                recon_eval_encoder=config['recon_eval_encoder'],
                device=config['device']
            )
        except Exception as e:
            print(f"[ERROR] {src} -> {trg}: {e}")
            continue

        for subj, r, layer_metrics in layer_results:
            for layer, acc in layer_metrics:
                result_data.append({
                    'Source': src,
                    'Target': trg,
                    'Identification accuracy': acc,
                    'Method': config['method'],
                    'ROI': r,
                    'Layer': layer
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
        'recon_eval_encoder': 'AlexNet',
        'device': 'cuda:0',
        'image_labels': [str(i) for i in range(1, 101)],
        'exclude_labels': ['1', '12', '23', '34', '45', '56', '67', '78', '89', '100'],
        'method': 'Content_loss',
        'output_file': 'attention_image_dnn_identification.csv',
        # Set to None to run all subject pairs.
        'example_pair': ('sub02', 'sub01')
    }

    result_data = run_all_pairs(config)
    save_results(result_data, config['output_file'])

    print('All done')


if __name__ == '__main__':
    main()
