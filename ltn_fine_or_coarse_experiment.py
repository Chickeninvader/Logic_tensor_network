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
from abc import ABC
from datetime import datetime
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from typing import List, Dict
import pickle
import random
from helper_function import *


def train(dataloader,
          base_model: FineTuner, logits_to_predicate,
          beta,
          epoch,
          optimizer,
          scheduler,
          loss_mode,
          fine_grain_only=False, mode='normal',
          device=torch.device('cpu'),
          coarse_label_dict={}, fine_label_dict={}, coarse_to_fine={}):
    """
    Train the model using the provided dataloader for one epoch.

    Args:
        dataloader (DataLoader): Dataloader for training data.
        base_model (FineTuner): The model to be trained.
        logits_to_predicate: Function to convert logits to predicates.
        beta (float): specify proportion of ltn and normal loss
        epoch (int): training iteration 
        fine_grain_only (bool): If True, train only on fine-grained labels.
        mode (str): Training mode: 'normal', 'ltn_normal', or 'ltn_combine'
        coarse_label_dict (dict, optional): Dictionary mapping coarse labels to numerical labels. Default is an empty dictionary.
        fine_label_dict (dict, optional): Dictionary mapping fine labels to numerical labels. Default is an empty dictionary.
        coarse_to_fine (dict, optional): Dictionary mapping coarse labels to corresponding fine labels. Default is an empty dictionary..

    Returns:
        float: Running loss.
        float: Precision for fine-grained labels.
        float: Recall for fine-grained labels.
        float: Precision for coarse labels.
        float: Recall for coarse labels.
    """
    num_coarse_label = len(coarse_label_dict)
    num_fine_label = len(fine_label_dict)
    num_all_label = num_fine_label + num_coarse_label

    base_model.train()
    size = len(dataloader)
    running_loss = 0.0

    fine_label_ground_truth = []
    fine_label_prediction = []
    coarse_label_ground_truth = []
    coarse_label_prediction = []

    with tqdm(total=size) as pbar:
        description = "Epoch " + str(epoch)
        pbar.set_description_str(description)

        for batch_idx, (data, labels_coarse, labels_fine, image_path) in enumerate(dataloader):
            # Zero gradient
            optimizer.zero_grad(set_to_none=True)

            # Put image to device
            data, labels_coarse, labels_fine = data.to(
                device), labels_coarse.to(device), labels_fine.to(device)

            # Get ground truth
            labels_coarse_one_hot = torch.nn.functional.one_hot(
                labels_coarse, num_classes=num_all_label).float()
            labels_fine_one_hot = torch.nn.functional.one_hot(
                labels_fine, num_classes=num_all_label).float()

            # make prediction
            prediction = base_model(data)

            # TODO: change ground truth depending on using coarse or fine mode
            if fine_grain_only:
                labels_one_hot = labels_fine_one_hot[:, num_coarse_label:]
            else:
                labels_one_hot = labels_coarse_one_hot[:, :num_coarse_label]

            loss = calculate_loss(mode, base_model, logits_to_predicate, prediction, labels_coarse, labels_fine,
                                  coarse_label_dict, fine_label_dict, coarse_to_fine, fine_grain_only,
                                  labels_one_hot, loss_mode, beta)

            running_loss += loss.item()

            # Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10.0)
            optimizer.step()
            running_loss += loss

            # Accuracy evaluation of coarse and fine grain
            prediction = prediction.cpu().detach()

            # TODO: change to get ground truth and prediction for find and coarse mode
            if fine_grain_only:
                prediction_fine_label = prediction
                fine_label_prediction_batch = torch.argmax(
                    prediction_fine_label, dim=1)
                fine_label_prediction.extend(fine_label_prediction_batch)
                fine_label_ground_truth.extend(
                    labels_fine.cpu().detach() - num_coarse_label)
            else:
                prediction_coarse_label = prediction
                coarse_label_prediction_batch = torch.argmax(
                    prediction_coarse_label, dim=1)
                coarse_label_prediction.extend(coarse_label_prediction_batch)
                coarse_label_ground_truth.extend(labels_coarse.cpu().detach())

            pbar.update()

        # Compute running loss
        running_loss = running_loss / size

        # Compute evaluation metrics
        # TODO: change code to display metric appropriately with fine and coarse mode
        if fine_grain_only:
            accuracy_coarse = 0
            precision_coarse = 0
            recall_coarse = 0
            accuracy_fine = accuracy_score(
                fine_label_ground_truth, fine_label_prediction, normalize=True)
            precision_fine = precision_score(
                fine_label_ground_truth, fine_label_prediction, average='macro')
            recall_fine = recall_score(
                fine_label_ground_truth, fine_label_prediction, average='macro')

        else:
            accuracy_fine = 0
            precision_fine = 0
            recall_fine = 0
            accuracy_coarse = accuracy_score(
                coarse_label_ground_truth, coarse_label_prediction, normalize=True)
            precision_coarse = precision_score(
                coarse_label_ground_truth, coarse_label_prediction, average='macro')
            recall_coarse = recall_score(
                coarse_label_ground_truth, coarse_label_prediction, average='macro')

        # print evaluation metric:

        pbar.set_postfix_str(" epoch %d | loss %.4f | Train coarse acc %.3f |Train coarse Prec %.3f | Train coarse Rec %.3f | Train fine acc %.3f |Train fine Prec %.3f | Train fine Rec %.3f" %
                             (epoch, running_loss, accuracy_coarse, precision_coarse, recall_coarse, accuracy_fine, precision_fine, recall_fine))

        # Update learning rate
        scheduler.step()

        save_metric = [float(running_loss.detach().to("cpu")),
                       accuracy_fine, precision_fine, recall_fine,
                       accuracy_coarse, precision_coarse, recall_coarse]

    return save_metric


@torch.no_grad()
def valid(dataloader,
          base_model, logits_to_predicate,
          beta,
          loss_mode,
          fine_grain_only=False, mode='normal',
          device=torch.device('cpu'),
          coarse_label_dict={}, fine_label_dict={}, coarse_to_fine={},):
    """
    Validate the model using the provided dataloader.

    Args:
        dataloader (DataLoader): Dataloader for validation data.
        base_model (FineTuner): The model to be evaluated.
        logits_to_predicate (function): Function to convert logits to predicates.
        beta: specify proportion of ltn and normal loss
        fine_grain_only (bool, optional): If True, validate only on fine-grained labels. Default is False.
        mode (str, optional): Validation mode: 'normal', 'ltn_normal', or 'ltn_combine'. Default is 'normal'.
        device (torch.device, optional): Device to perform computations on. Default is 'cuda'.
        coarse_label_dict (dict, optional): Dictionary mapping coarse labels to numerical labels. Default is an empty dictionary.
        fine_label_dict (dict, optional): Dictionary mapping fine labels to numerical labels. Default is an empty dictionary.
        coarse_to_fine (dict, optional): Dictionary mapping coarse labels to corresponding fine labels. Default is an empty dictionary.
        model_name (string, optional): Name of the model to save, default is empty string

    Returns:
        float: Running loss.
        float: Precision for fine-grained labels.
        float: Recall for fine-grained labels.
        float: Precision for coarse labels.
        float: Recall for coarse labels.
    """
    num_coarse_label = len(coarse_label_dict)
    num_fine_label = len(fine_label_dict)
    num_all_label = num_fine_label + num_coarse_label
    base_model.eval()
    size = len(dataloader)
    running_loss = 0.0
    loss_fc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    epoch = scheduler.last_epoch
    fine_label_ground_truth = []
    fine_label_prediction = []
    coarse_label_ground_truth = []
    coarse_label_prediction = []

    with tqdm(total=size) as pbar:
        description = "Evaluate test set: "
        pbar.set_description_str(description)

        for batch_idx, (data, labels_coarse, labels_fine, image_path) in enumerate(dataloader):

            # Put image to device
            data, labels_coarse, labels_fine = data.to(
                device), labels_coarse.to(device), labels_fine.to(device)

            # Get ground truth
            labels_coarse_one_hot = torch.nn.functional.one_hot(
                labels_coarse, num_classes=num_all_label).float()
            labels_fine_one_hot = torch.nn.functional.one_hot(
                labels_fine, num_classes=num_all_label).float()

            # make prediction
            prediction = base_model(data)

            # TODO: change ground truth depending on using coarse or fine mode
            if fine_grain_only:
                labels_one_hot = labels_fine_one_hot[:, num_coarse_label:]
            else:
                labels_one_hot = labels_coarse_one_hot[:, :num_coarse_label]

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

            if mode == 'ltn_normal':
                sat_agg = compute_sat_normally(base_model, logits_to_predicate,
                                               prediction, labels_coarse, labels_fine,
                                               coarse_label_dict, fine_label_dict, coarse_to_fine,
                                               fine_grain_only)
                loss = 1. - sat_agg

            if mode == 'ltn_combine':
                sat_agg = compute_sat_normally(base_model, logits_to_predicate,
                                               prediction, labels_coarse, labels_fine,
                                               coarse_label_dict, fine_label_dict, coarse_to_fine,
                                               fine_grain_only)
                if loss_mode == 'cross_entropy':
                    loss_fc = torch.nn.CrossEntropyLoss()
                    loss = beta*(1. - sat_agg) + (1 - beta) * \
                        (loss_fc(prediction, labels_one_hot))

                elif loss_mode == 'marginal':
                    loss_fc = torch.nn.MultiLabelMarginLoss()
                    loss = beta*(1. - sat_agg) + (1 - beta) * \
                        (loss_fc(prediction, labels_one_hot.long()))

                elif loss_mode == 'softmarginal':
                    loss_fc = torch.nn.MultiLabelSoftMarginLoss()
                    loss = beta*(1. - sat_agg) + (1 - beta) * \
                        (loss_fc(prediction, labels_one_hot))

            running_loss += loss.item()

            running_loss += loss.item()

            # Accuracy evaluation of coarse and fine grain
            prediction = prediction.cpu().detach()

            # TODO: change to get ground truth and prediction for find and coarse mode
            if fine_grain_only:
                prediction_fine_label = prediction
                fine_label_prediction_batch = torch.argmax(
                    prediction_fine_label, dim=1)
                fine_label_prediction.extend(fine_label_prediction_batch)
                fine_label_ground_truth.extend(
                    labels_fine.cpu().detach() - num_coarse_label)
            else:
                prediction_coarse_label = prediction
                coarse_label_prediction_batch = torch.argmax(
                    prediction_coarse_label, dim=1)
                coarse_label_prediction.extend(coarse_label_prediction_batch)
                coarse_label_ground_truth.extend(labels_coarse.cpu().detach())

            pbar.update()

        # Compute running loss
        running_loss = running_loss / size

        # Compute evaluation metrics
        if fine_grain_only:
            accuracy_coarse = 0
            precision_coarse = 0
            recall_coarse = 0
            accuracy_fine = accuracy_score(
                fine_label_ground_truth, fine_label_prediction, normalize=True)
            precision_fine = precision_score(
                fine_label_ground_truth, fine_label_prediction, average='macro')
            recall_fine = recall_score(
                fine_label_ground_truth, fine_label_prediction, average='macro')

        else:
            accuracy_fine = 0
            precision_fine = 0
            recall_fine = 0
            accuracy_coarse = accuracy_score(
                coarse_label_ground_truth, coarse_label_prediction, normalize=True)
            precision_coarse = precision_score(
                coarse_label_ground_truth, coarse_label_prediction, average='macro')
            recall_coarse = recall_score(
                coarse_label_ground_truth, coarse_label_prediction, average='macro')

        # print the training metrics

        pbar.set_postfix_str(" epoch %d | loss %.4f | Train coarse acc %.3f |Train coarse Prec %.3f | Train coarse Rec %.3f | Train fine acc %.3f |Train fine Prec %.3f | Train fine Rec %.3f" %
                             (epoch, running_loss, accuracy_coarse, precision_coarse, recall_coarse, accuracy_fine, precision_fine, recall_fine))

        save_metric = [running_loss,
                       accuracy_fine, precision_fine, recall_fine,
                       accuracy_coarse, precision_coarse, recall_coarse]

    return save_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train and evaluate the model')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path containing all files including results, models, train, and test data')
    parser.add_argument('--mode', choices=["normal", "ltn_normal", "ltn_combine"], default='ltn_combine',
                        help='Training mode: normal, ltn_normal, or ltn_combine')
    parser.add_argument('--vit_model_index', choices=[0, 1, 2, 3, 4], type=int, default=0,
                        help='Index of the VIT model to use, including b-16, b-32, l-16, l-32, h-14')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='Beta value for the loss function')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('--fine_grain_only', action='store_true',
                        help='Specify to train fine_grain only. Default is False, which trains also coarse label.')
    parser.add_argument('--loss_mode', choices=['cross_entropy', 'marginal', 'softmarginal'], default='cross_entropy',
                        help='Choose the appropriate loss mode (binary or softmarginal). Default is softmarginal.')
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='Load checkpoint or train from scratch')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')

    args = parser.parse_args()

    # Assigning argparse values to variables
    base_path = args.base_path
    fine_grain_only = args.fine_grain_only
    description_label = 'fine' if fine_grain_only else 'coarse'
    description = 'model ' + str(args.vit_model_index) + \
        ' ' + args.mode + ' ' + args.loss_mode + \
        ' ' + str(args.lr) + ' ' + str(args.beta) + ' ' + description_label
    print(description)
    mode = args.mode
    vit_model_index = args.vit_model_index
    beta = args.beta
    lr = args.lr
    fine_grain_only = args.fine_grain_only
    loss_mode = args.loss_mode
    load_checkpoint = args.load_checkpoint
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # All label
    category_dict = {
        'Air Defence': ['30N6E', 'Iskander', 'Pantsir-S1', 'Rs-24'],
        'BMP': ['BMP-1', 'BMP-2', 'BMP-T15'],
        'BTR': ['BRDM', 'BTR-60', 'BTR-70', 'BTR-80'],
        'Tank': ['T-14', 'T-62', 'T-64', 'T-72', 'T-80', 'T-90'],
        'SPA': ['2S19_MSTA', 'BM-30', 'D-30', 'Tornado', 'TOS-1'],
        'BMD': ['BMD'],
        'MT_LB': ['MT_LB']
    }

    coarse_label_dict, fine_label_dict, coarse_to_fine = create_label_dict(
        category_dict)

    # Print the resulting dictionaries
    print("coarse_label_dict:")
    print(coarse_label_dict)
    print("\nfine_label_dict:")
    print(fine_label_dict)
    print("\ncoarse_to_fine:")
    print(coarse_to_fine)

    l = create_one_hot_tensors(
        fine_label_dict, coarse_label_dict, fine_grain_only)
    inverse_dict = create_inverse_dict(coarse_label_dict, fine_label_dict)

    # Constants and Configuration
    image_resize = 224
    num_coarse_label = len(coarse_label_dict)
    num_fine_label = len(fine_label_dict)
    num_all_label = num_fine_label + num_coarse_label
    num_output = num_fine_label if fine_grain_only else num_coarse_label
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)
    base_train_folder = f"{base_path}/dataset/train"
    base_test_folder = f"{base_path}/dataset/test"
    load_checkpoint_path = f"{base_path}/model/model_{description}.pth"

    # Load dataset
    df_train, df_test = process_image_folders(
        base_train_folder, base_test_folder, coarse_label_dict, fine_label_dict)
    train_loader, test_loader = create_data_loaders(
        df_train, df_test, image_resize, batch_size, num_coarse_label, num_all_label)

    print('get dataset successfully')

    # Model Initialization
    base_model = VITFineTuner(vit_model_index, num_output).to(device)
    logits_to_predicate = ltn.Predicate(LogitsToPredicate()).to(ltn.device)
    print('model initialization successfully')

    # Training Configuration
    optimizer = torch.optim.Adam(base_model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

    # Training Loop
    evaluation_metric_train = []
    evaluation_metric_valid = []

    if load_checkpoint:
        # Load the checkpoint
        checkpoint = torch.load(load_checkpoint_path)

        # Load model and optimizer states
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state, if available in the checkpoint
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

            # Retrieve the epoch information if available in the checkpoint
            loaded_epoch = scheduler.last_epoch

        # Restoring evaluation_metric_train.
        with open(f'{base_path}/model/evaluation_metric_train_{description}.pkl', 'rb') as f:
            evaluation_metric_train = pickle.load(f)

        # Restoring evaluation_metric_valid
        with open(f'{base_path}/model/evaluation_metric_valid_{description}.pkl', 'rb') as f:
            evaluation_metric_valid = pickle.load(f)

        print('load checkpoint successfully')

    else:
        print('train from beginning')
        loaded_epoch = 0  # If not loading a checkpoint, start training from epoch 0

    max_accuracy = 0

    for epoch in range(loaded_epoch, num_epochs):
        with ClearCache(device):
            evaluation_metric_train.append(train(train_loader,
                                                 base_model, logits_to_predicate,
                                                 beta,
                                                 epoch,
                                                 optimizer,
                                                 scheduler,
                                                 loss_mode,
                                                 fine_grain_only, mode,
                                                 device,
                                                 coarse_label_dict, fine_label_dict, coarse_to_fine))
            evaluation_metric_valid.append(valid(test_loader,
                                                 base_model, logits_to_predicate,
                                                 beta,
                                                 loss_mode,
                                                 fine_grain_only, mode,
                                                 device,
                                                 coarse_label_dict, fine_label_dict, coarse_to_fine))

            # TODO: Checkpoint best and last epoch
            # Get best accuracy (index 1 and 4) as the evaluation metric for checkpoint
            # Save the model if the validation metrics improve

            torch.save({"model_state_dict": base_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                       f"{base_path}/model/model_{description}.pth")

            print(f"Saved PyTorch Model State to {description}")

            # Saving evaluation_metric_train
            with open(f'{base_path}/model/evaluation_metric_train_{description}.pkl', 'wb') as f:
                pickle.dump(evaluation_metric_train, f)

            # Saving evaluation_metric_valid
            with open(f'{base_path}/model/evaluation_metric_valid_{description}.pkl', 'wb') as f:
                pickle.dump(evaluation_metric_valid, f)

        print('#' * 100)

    # TODO: change the directory to save result
    # Save evaluation metrics to the result folder with the description

    # Create a folder with the name 'description' inside the 'result' folder
    result_folder_path = os.path.join(base_path, "result", description)
    os.makedirs(result_folder_path, exist_ok=True)

    save_evaluation_metric(evaluation_metric_train,
                           evaluation_metric_valid, result_folder_path, description)

    # Save confusion matrices to the result folder with the description
    save_confusion_matrices(num_coarse_label, num_all_label,
                            base_model, test_loader, result_folder_path,
                            fine_grain_only, device, coarse_label_dict, fine_label_dict, description)
