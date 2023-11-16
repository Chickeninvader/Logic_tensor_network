# This file will contain all helper function for training fine/coarse grain image only and both coarse + fine grain

# import library
import glob
import argparse
import pandas as pd
import torch
import cv2
import skimage
import torchvision
import torchsummary
from skimage import io, transform
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import re
import requests
from torchvision import datasets
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
import ltn
from torchsummary import summary
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score
from typing import Tuple
from time import time
from pathlib import Path
import sys
import timm
from abc import ABC
from datetime import datetime
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from typing import List, Dict
import pickle
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler


def create_label_dict(category_dict):
    # Initialize dictionaries
    coarse_label_dict = {}
    fine_label_dict = {}
    coarse_to_fine = {}

    # Assign numerical labels
    coarse_label_counter = 0
    fine_label_counter = len(category_dict)

    # Iterate through the input dictionary
    for category, labels in category_dict.items():
        # Assign a numerical label to the coarse category
        coarse_label_dict[category] = coarse_label_counter

        # Create an empty list to store fine labels for this coarse category
        coarse_to_fine[coarse_label_counter] = []

        # Iterate through labels in the category
        for label in labels:
            # Assign a numerical label to the fine label
            fine_label_dict[label] = fine_label_counter

            # Add the fine label to the list of fine labels for this coarse category
            coarse_to_fine[coarse_label_counter].append(fine_label_counter)

            # Increment the fine label counter
            fine_label_counter += 1

        # Increment the coarse label counter
        coarse_label_counter += 1

    # Return the resulting dictionaries
    return coarse_label_dict, fine_label_dict, coarse_to_fine


def create_one_hot_tensors(fine_label_dict, coarse_label_dict):
    l = {}
    num_labels = len(fine_label_dict) + len(coarse_label_dict)
    for label in range(num_labels):
        one_hot = torch.zeros(num_labels)
        one_hot[label] = 1.0
        l[label] = ltn.Constant(one_hot, trainable=True)
    return l


def create_inverse_dict(coarse_label_dict, fine_label_dict):
    inverse_dict = {}
    for label, value in coarse_label_dict.items():
        inverse_dict[value] = label

    for label, value in fine_label_dict.items():
        inverse_dict[value] = label

    return inverse_dict


def extract_labels(folder_path):
    parts = folder_path.split(os.path.sep)
    coarse_label = parts[-2]
    fine_label = parts[-1]
    return coarse_label, fine_label


def search_for_images_and_labels(folder):
    data = []
    for image_path in glob.glob(os.path.join(folder, "*.jpg")):
        coarse_label, fine_label = extract_labels(folder)
        data.append({
            'completed_relative_path': os.path.abspath(image_path),
            'Coarse label': coarse_label,
            'fine label': fine_label
        })
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            data.extend(search_for_images_and_labels(subfolder_path))
    return data


def process_image_folders(base_train_folder, base_test_folder):
    # Process train folder
    train_data = search_for_images_and_labels(base_train_folder)
    df_train = pd.DataFrame(train_data)
    df_train['Coarse label'] = df_train['Coarse label'].replace(
        coarse_label_dict)
    df_train['fine label'] = df_train['fine label'].replace(fine_label_dict)

    # Filter train dataset
    coarse_train_labels = [label for _, label in coarse_label_dict.items()]
    fine_train_labels = [label for _, label in fine_label_dict.items()]
    filter_train_coarse = df_train['Coarse label'].isin(coarse_train_labels)
    filter_train_fine = df_train['fine label'].isin(fine_train_labels)
    df_train = df_train[filter_train_coarse &
                        filter_train_fine].reset_index(drop=True)

    # Process test folder
    test_data = search_for_images_and_labels(base_test_folder)
    df_test = pd.DataFrame(test_data)
    df_test['Coarse label'] = df_test['Coarse label'].replace(
        coarse_label_dict)
    df_test['fine label'] = df_test['fine label'].replace(fine_label_dict)

    # Filter test dataset
    coarse_test_labels = [label for _, label in coarse_label_dict.items()]
    fine_test_labels = [label for _, label in fine_label_dict.items()]
    filter_test_coarse = df_test['Coarse label'].isin(coarse_test_labels)
    filter_test_fine = df_test['fine label'].isin(fine_test_labels)
    df_test = df_test[filter_test_coarse &
                      filter_test_fine].reset_index(drop=True)

    return df_train, df_test


class DatasetGenerator():
    """
    Create a dataloader to efficiently get data. The argument include:
        - dataset: the dataframe containing image_path and label
        - image_resize: size of the image
    """

    def __init__(self, dataset, image_resize):
        self.dataset = dataset
        self.image_resize = image_resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get index
        idx = index % len(self.dataset)
        image_path = self.dataset['completed_relative_path'][idx]
        image = Image.open(image_path)
        image_rgb = Image.new("RGB", image.size)
        image_rgb.paste(image)

        coarse_label = self.dataset['Coarse label'][idx]
        fine_label = self.dataset['fine label'][idx]

        # Change image to float, resize image and

        imagenet_stats = ([0.5] * 3, [0.5] * 3)
        preprocess = transforms.Compose([
            transforms.Resize((self.image_resize, self.image_resize)),
            transforms.RandomResizedCrop(
                max((self.image_resize, self.image_resize))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats)
        ])

        image_rgb = preprocess(image_rgb)

        return image_rgb, coarse_label, fine_label, image_path


def create_data_loaders(df_train, df_test, image_resize, batch_size, num_coarse_label, num_all_label):
    """
    Create data loaders for the training and testing datasets.

    Args:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        image_resize (int): Size to which images will be resized.
        batch_size (int): Number of samples in each batch.
        num_coarse_label (int): Number of coarse labels.
        num_all_label (int): Total number of labels including fine and coarse labels.

    Returns:
        DataLoader: Training data loader.
        DataLoader: Testing data loader.
    """
    train_dataset = DatasetGenerator(df_train, image_resize)
    test_dataset = DatasetGenerator(df_test, image_resize)

    # Compute class weights for weighted sampling
    fine_distribution = df_train["fine label"].value_counts().tolist()
    class_weights = [1 / df_train["fine label"].value_counts()[i]
                     for i in range(num_coarse_label, num_all_label)]
    class_weights = [0] * num_coarse_label + class_weights
    image_weights = [class_weights[i] for i in df_train['fine label']]
    weight_sampler = torch.utils.data.WeightedRandomSampler(
        image_weights, len(df_train))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=4, pin_memory=True, sampler=weight_sampler)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, test_loader


class ClearCache:
    def __init__(self, device: torch.device):
        self.device_backend = {'cuda': torch.cuda,
                               'cpu': None}[device]

    def __enter__(self):
        if self.device_backend:
            self.device_backend.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device_backend:
            self.device_backend.empty_cache()


class FineTuner(torch.nn.Module, ABC):
    def __str__(self) -> str:
        return self.__class__.__name__.split('Fine')[0].lower()

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())

# TODO: Whenever there is the change in loss function, check the implementation accordingly, to whether include
# softmax or sigmoid on classifier or not


class VITFineTuner(FineTuner):
    def __init__(self,
                 vit_model_index: int,
                 num_classes: int):
        super().__init__()
        vit_model_name = ['b_16',
                          'b_32',
                          'l_16',
                          'l_32',
                          'h_14']
        self.vit_model_name = vit_model_name[vit_model_index]
        vit_model = eval(f'torchvision.models.vit_{self.vit_model_name}')
        vit_weights = eval(f"torchvision.models.ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('_')])}"
                           f"_Weights.DEFAULT")
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x

    def __str__(self):
        return f'{super().__str__()}_{self.vit_model_name}'

# TODO: Whenever there is the change in loss function, check the implementation accordingly, to whether include
# softmax or sigmoid on classifier or not


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label d. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class d.
    """

    def __init__(self):
        super(LogitsToPredicate, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, d):
        probs = self.sigmoid(x)
        out = torch.sum(probs * d, dim=1)
        return out


def compute_sat_normally(base_model,
                         logits_to_predicate,
                         data, labels_coarse, labels_fine,
                         coarse_label_dict, fine_label_dict,
                         coarse_to_fine, fine_grain_only=False, train_mode=False):
    """
    compute satagg function for rules
    argument:
      - base_model: get probability of the class
      - logits_to_predicate: get the satisfaction of a variable given the label
      - data, labels_coarse, labels_fine
      - coarse_label_dict, fine_label_dict,
      - coarse_to_fine
      - fine_grain_only: if true, the sat is changed accordingly
      - train: whether to train model again, when data is still not convert to prediction yet

    return:
      sat_agg: sat_agg for all the rules

    """
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    if train_mode:
        prediction = base_model(data)
    else:
        prediction = data

    x = ltn.Variable("x", prediction)

    x_variables = {}
    for name, label in fine_label_dict.items():
        x_variables[label] = ltn.Variable(
            name, prediction[labels_fine == label])
    for name, label in coarse_label_dict.items():
        x_variables[label] = ltn.Variable(
            name, prediction[labels_coarse == label])

    sat_agg_list = []
    sat_agg_label = []

    # Coarse labels: for all x[i], x[i] -> l[i]

    for i in coarse_label_dict.values():
        if x_variables[i].value.numel() != 0:
            sat_agg_label.append(
                f'for all (coarse label) x[{i}], x[{i}] -> l[{i}]')
            sat_agg_list.append(
                Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i])))

    # Coarse Label: for all x[coarse], - {x[coarse] and x[different coarse]}
    # TODO: Double check the rule

    for i in coarse_label_dict.values():
        for j in coarse_label_dict.values():
            if i != j:
                sat_agg_label.append(
                    f'for all (coarse label) x[{i}], - (x[{i}] ^ x[{j}])')
                sat_agg_list.append(
                    Forall(x, Not(And(logits_to_predicate(x, l[i]), logits_to_predicate(x, l[j])))))

    # Fine to coarse label: for all x[fine], x[fine] and x[correspond coarse]

    for label_coarse, label_fine_list in coarse_to_fine.items():
        for label_fine in label_fine_list:
            if x_variables[label_fine].value.numel() != 0:
                sat_agg_label.append(
                    f'for all (fine label) x[{label_fine}], (x[{label_fine}] ^ x[{label_coarse}])')
                sat_agg_list.append(Forall(x_variables[label_fine],
                                           And(logits_to_predicate(x_variables[label_fine], l[label_fine]), logits_to_predicate(x_variables[label_fine], l[label_coarse])))
                                    )

    # Fine labels: for all x[i], x[i] -> l[i]

    for i in fine_label_dict.values():
        if x_variables[i].value.numel() != 0:
            sat_agg_label.append(
                f'for all (fine label) x[{i}], x[{i}] -> l[{i}]')
            sat_agg_list.append(
                Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i])))

    # TODO: Double check the rule
    # Fine Label: for all x[fine], -{x[fine], x[diff_fine]}

    for _, label_fine_list in coarse_to_fine.items():
        for label_fine in label_fine_list:
            for i in label_fine_list:
                if (x_variables[label_fine].value.numel() != 0) and (i != label_fine):
                    sat_agg_label.append(
                        f'for all (fine label) x[{label_fine}], -(x[{label_fine}] -> l[{i}])')
                    sat_agg_list.append(Forall(x_variables[label_fine],
                                        Not(logits_to_predicate(x_variables[label_fine], l[i]))))
    sat_agg = SatAgg(
        *sat_agg_list
    )
    return sat_agg


def transform_evaluation_metric(metric_list):
    transformed_metrics = []
    for metric_dict in metric_list:
        try:
            transformed_metrics.append({
                'running_loss': metric_dict[0],
                'accuracy_fine': metric_dict[1],
                'precision_fine': metric_dict[2],
                'recall_fine': metric_dict[3],
                'accuracy_coarse': metric_dict[4],
                'precision_coarse': metric_dict[5],
                'recall_coarse': metric_dict[6]
            })
        except:
            print('error in getting some metric')
    return transformed_metrics


def save_evaluation_metric(evaluation_metric_train_raw, evaluation_metric_valid_raw, path: str, description):
    """
    Save the evaluation metric plot.

    Args:
        path (str): File path to save the plot.
        evaluation_metric_train (list): List of dictionaries containing evaluation metrics for training data.
        evaluation_metric_valid (list): List of dictionaries containing evaluation metrics for validation data.
        description (str)

    Returns:
        None
    """

    num_epochs = len(evaluation_metric_train_raw)
    evaluation_metric_train = transform_evaluation_metric(
        evaluation_metric_train_raw)
    evaluation_metric_valid = transform_evaluation_metric(
        evaluation_metric_valid_raw)
    y_limits = [0.0, 1.0]

    for metric in ['running_loss', 'accuracy_fine', 'precision_fine', 'recall_fine', 'accuracy_coarse', 'precision_coarse', 'recall_coarse']:

        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), [
                 element[metric] for element in evaluation_metric_train], label='Training', color='green')
        plt.plot(range(num_epochs), [
                 element[metric] for element in evaluation_metric_valid], label='Validation', color='blue')
        plt.ylim(y_limits)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.legend()
        plt.grid(True)

        save_path = f'{path}/{description}_{metric.capitalize()}.png'

        plt.savefig(save_path)
        plt.close()  # Close the plot to clear the memory


def calculate_metrics_per_label(y_true: List[int], y_pred: List[int],
                                labels: List[int]) -> Tuple[List[float], List[float], List[float], List[List[int]]]:
    """
    Calculates precision, recall, F1 score, and confusion matrix for each label.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        labels (List[int]): List of label indices.

    Returns:
        Tuple[List[float], List[float], List[float], List[List[int]]]: Precision, recall, F1 score, and confusion matrix.

    """
    # accuracy_per_label = accuracy_score(y_true, y_pred)
    precision_per_label = precision_score(
        y_true, y_pred, average=None, labels=labels)
    recall_per_label = recall_score(
        y_true, y_pred, average=None, labels=labels)
    accuracy_per_label = [precision * recall for precision,
                          recall in zip(precision_per_label, recall_per_label)]
    f1_per_label = f1_score(y_true, y_pred, average=None, labels=labels)
    confusion_mat = confusion_matrix(y_true, y_pred, labels=labels)

    return accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, confusion_mat


def save_metrics_to_excel(y_true: List[int], y_pred: List[int],
                          label_dict: Dict[int, str],
                          model_name: str, path: str, description: str) -> None:
    """
    Calculates metrics per label and saves the results to an Excel file.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        label_dict (Dict[int, str]): Dictionary mapping label indices to label names.
        model_name (str): Name of the model.
        path (str): Directory where the Excel file will be saved.
        description (str)
    """
    label_temp = [i for i in label_dict.values()]
    accuracy, precision, recall, f1, confusion = calculate_metrics_per_label(
        y_true, y_pred, label_temp)
    metrics_df = pd.DataFrame(columns=['Label', 'Accuracy', 'Precision', 'Recall',
                              'F1', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
    inverse_dict = {value: key for key, value in label_dict.items()}

    for label_idx, label in enumerate(label_temp):
        metrics_df = metrics_df.append({
            'Label': inverse_dict[int(label)],
            'Accuracy': accuracy[label_idx],
            'Precision': precision[label_idx],
            'Recall': recall[label_idx],
            'F1': f1[label_idx],
            'True Positives': confusion[label_idx][label_idx],
            'True Negatives': confusion.sum() - confusion[label_idx].sum() - confusion[:, label_idx].sum() + confusion[label_idx][label_idx],
            'False Positives': confusion[:, label_idx].sum() - confusion[label_idx][label_idx],
            'False Negatives': confusion[label_idx].sum() - confusion[label_idx][label_idx]
        }, ignore_index=True)

    metrics_df.to_excel(
        f'{path}/{description}_coarse_grained_{model_name}_test_metric.xlsx', index=False)


def save_confusion_matrices(num_coarse_label: int, num_all_labels: int,
                            base_model, test_loader: DataLoader,
                            save_path: str,
                            fine_grain_only: bool,
                            description: str,) -> None:
    """
    Compute and save confusion matrices for coarse and fine labels based on the predictions
    from the provided base_model and test_loader. Save the generated matrices as images.

    Args:
        num_coarse_label (int): Number of coarse labels.
        num_all_labels (int): Total number of labels (including coarse and fine labels).
        base_model: PyTorch model for prediction.
        test_loader (DataLoader): DataLoader containing test data.
        save_path (str): Path to save the generated confusion matrix images.
        fine_grain_only (bool): Train fine grain only or not
        description (str)

    Returns:
        None
    """
    coarse_index = slice(num_coarse_label)
    fine_index = slice(num_coarse_label, num_all_labels)

    coarse_label_ground_truth = []
    coarse_label_prediction = []
    fine_label_ground_truth = []
    fine_label_prediction = []
    image_path_list = []

    print("Save confusion matrices")

    # Iterate through the test data and make predictions
    for batch_idx, (data, labels_coarse, labels_fine, image_path) in enumerate(test_loader):
        data = data.to(device)

        prediction = base_model(data).cpu().detach()

        prediction_coarse_label = prediction[:, coarse_index]
        coarse_label_prediction_batch = torch.argmax(
            prediction_coarse_label, dim=1)
        coarse_label_prediction.extend(coarse_label_prediction_batch)
        coarse_label_ground_truth.extend(labels_coarse)

        prediction_fine_label = prediction[:, fine_index]
        fine_label_prediction_batch = torch.argmax(
            prediction_fine_label, dim=1) + num_coarse_label
        fine_label_prediction.extend(fine_label_prediction_batch)
        fine_label_ground_truth.extend(labels_fine)

        image_path_list.extend(image_path)

    # Compute confusion matrix for coarse labels
    confusion_matrix_coarse = metrics.confusion_matrix(
        coarse_label_ground_truth, coarse_label_prediction)
    display_labels_coarse = [str(label)
                             for label in range(num_coarse_label)]

    # Plot and save coarse label confusion matrix
    fig_coarse, ax_coarse = plt.subplots(figsize=(15, 15))
    cm_display_coarse = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_coarse,
                                               display_labels=display_labels_coarse)
    cm_display_coarse.plot(ax=ax_coarse, values_format='d')
    ax_coarse.set_title('Coarse Label Confusion Matrix')
    plt.savefig(
        f'{save_path}/{description}_coarse_label_confusion_matrix.png')
    plt.close(fig_coarse)

    print('Saved coarse label confusion matrix successfully')

    # Compute confusion matrix for fine labels
    confusion_matrix_fine = metrics.confusion_matrix(
        fine_label_ground_truth, fine_label_prediction)
    display_labels_fine = [str(label) for label in range(
        num_coarse_label, num_all_labels)]

    # Plot and save fine label confusion matrix
    fig_fine, ax_fine = plt.subplots(figsize=(15, 15))
    cm_display_fine = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_fine,
                                             display_labels=display_labels_fine)
    cm_display_fine.plot(ax=ax_fine, values_format='d')
    ax_fine.set_title('Fine Label Confusion Matrix')
    plt.savefig(f'{save_path}/{description}_fine_label_confusion_matrix.png')
    plt.close(fig_fine)

    print('Saved fine label confusion matrix successfully')

    print('save coarse grain excel file')
    save_metrics_to_excel(coarse_label_ground_truth, coarse_label_prediction,
                          coarse_label_dict, base_model,
                          save_path,
                          description + '_coarse'
                          )

    print('save fine grain excel file')
    save_metrics_to_excel(fine_label_ground_truth, fine_label_prediction,
                          fine_label_dict, base_model,
                          save_path,
                          description + '_fine'
                          )

    print('save excel file successfully')


def calculate_loss(mode, base_model, logits_to_predicate, prediction, labels_coarse, labels_fine,
                   coarse_label_dict, fine_label_dict, coarse_to_fine, fine_grain_only,
                   labels_one_hot=None, loss_mode='cross_entropy', beta=0.8):
    """
    Calculate the loss based on the specified mode and loss function.

    Parameters:
    - mode (str): The training mode ('normal', 'ltn_normal', or 'ltn_combine').
    - base_model (torch.nn.Module): The base model.
    - logits_to_predicate (ltn.Predicate): The predicate for converting logits to truth values.
    - prediction (torch.Tensor): The model's prediction.
    - labels_coarse (torch.Tensor): Coarse-grained ground truth labels.
    - labels_fine (torch.Tensor): Fine-grained ground truth labels.
    - coarse_label_dict (dict): Mapping of coarse labels to indices.
    - fine_label_dict (dict): Mapping of fine labels to indices.
    - coarse_to_fine (dict): Mapping of coarse labels to corresponding fine labels.
    - fine_grain_only (bool): Specify whether to train fine-grained only.
    - labels_one_hot (torch.Tensor, optional): One-hot encoded ground truth labels.
    - loss_mode (str): The loss function to use ('cross_entropy', 'marginal', or 'softmarginal').
    - beta (float): The weight for the SAT loss.

    Returns:
    - torch.Tensor: The calculated loss.
    """
    if mode == 'normal':
        if loss_mode == 'cross_entropy':
            loss_fc = torch.nn.CrossEntropyLoss()
            loss = loss_fc(prediction, labels_one_hot)
        elif loss_mode == 'marginal':
            loss_fc = torch.nn.MultiLabelMarginLoss()
            loss = loss_fc(prediction, labels_one_hot.long())
        elif loss_mode == 'softmarginal':
            loss_fc = torch.nn.MultiLabelSoftMarginLoss()
            loss = loss_fc(prediction, labels_one_hot)

    elif mode == 'ltn_normal':
        # Compute SAT loss normally
        sat_agg = compute_sat_normally(base_model, logits_to_predicate,
                                       prediction, labels_coarse, labels_fine,
                                       coarse_label_dict, fine_label_dict, coarse_to_fine,
                                       fine_grain_only)
        # Calculate loss
        loss = 1. - sat_agg

    elif mode == 'ltn_combine':
        # Compute SAT loss normally
        sat_agg = compute_sat_normally(base_model, logits_to_predicate,
                                       prediction, labels_coarse, labels_fine,
                                       coarse_label_dict, fine_label_dict, coarse_to_fine,
                                       fine_grain_only)
        if loss_mode == 'cross_entropy':
            # Use CrossEntropyLoss for combined loss
            loss_fc = torch.nn.CrossEntropyLoss()
            loss = beta * (1. - sat_agg) + (1 - beta) * \
                (loss_fc(prediction, labels_one_hot))

        elif loss_mode == 'marginal':
            # Use MultiLabelMarginLoss for combined loss
            loss_fc = torch.nn.MultiLabelMarginLoss()
            loss = beta * (1. - sat_agg) + (1 - beta) * \
                (loss_fc(prediction, labels_one_hot.long()))

        elif loss_mode == 'softmarginal':
            # Use MultiLabelSoftMarginLoss for combined loss
            loss_fc = torch.nn.MultiLabelSoftMarginLoss()
            loss = beta * (1. - sat_agg) + (1 - beta) * \
                (loss_fc(prediction, labels_one_hot))

    return loss
