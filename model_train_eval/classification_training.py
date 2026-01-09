#!/usr/bin/env python3
"""
Multi-class Image Classification Training and Validation
==================================================

A training pipeline for classifying images into four categories:
bat, bird, human, and other.
The pipeline implements a ResNet architecture optimised for small input images.

Key Features
------------
- Stratified Group K-Fold cross-validation to prevent site-level data leakage
- Optional CIFAR-10 pre-training for backbone weight initialisation
- Class weighting to handle imbalanced datasets
- Flexible threshold calibration strategies for the target class (bat)

Architecture
------------
The model uses a CIFAR-style ResNet with:
- No aggressive downsampling in the stem (suitable for small inputs)
- Squeeze-and-Excitation blocks for channel attention (Hu et al., 2018)
- Global average pooling followed by dropout regularisation
- Single softmax classification head

References
----------
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
  with Warm Restarts. ICLR.

Usage
-----
With configuration file:
    $ python classification_training.py --config config.ini

With CLI arguments (overrides config file):
    $ python classification_training.py --data_dir /path/to/data --output_dir /path/to/output

Mixed usage (CLI arguments override config file values):
    $ python classification_training.py --config config.ini --epochs 50

Output
------
- Trained model weights (.keras format)
- Cross-validation metrics and confusion matrices
- Threshold calibration analysis plots
- Classification reports with standard deviations across folds

Date: 17/12/2025
"""

import os
import sys
import argparse
import configparser
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, fbeta_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Class definitions - order is significant for label encoding
CLASS_NAMES = ['1_bat', '2_bird', '3_human', '4_other']
CLASS_LABELS = list(range(len(CLASS_NAMES)))  # Integer labels: 0, 1, 2, 3
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
SHORT_NAMES = ['bat', 'bird', 'human', 'other']  # Abbreviated names for plots
BAT_ID = NAME_TO_ID['1_bat']  # Primary class of interest

# Training configuration flags
USE_CLASS_WEIGHTS = True          # Apply inverse frequency class weighting
BAT_WEIGHT_MULTIPLIER = 2.0       # Additional emphasis on bat class
PRETRAIN_ON_CIFAR10 = True        # Pre-train backbone on CIFAR-10

# CIFAR-10 pre-training hyperparameters
CIFAR_PRETRAIN_EPOCHS = 15
CIFAR_PRETRAIN_BATCH_SIZE = 256
CIFAR_PRETRAIN_VAL_SPLIT = 0.1
CIFAR_PRETRAIN_LR = 1e-3

# Fine-tuning hyperparameters
FINE_TUNE_LR = 1e-4               # Learning rate for main training
HEAD_START_EPOCHS = 5             # Epochs to train only classification head
HEAD_START_LR = 3e-4              # Learning rate for head-only training


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def extract_site_from_filename(filename):
    """
    Extract site identifier from standardised filename format.
    
    Expected filename format:
        <frame>_<tracking_id>_<site>_<date>_<camera>_clip_<clip_id>.png
    
    Parameters
    ----------
    filename : str
        Image filename following the standardised naming convention.
    
    Returns
    -------
    str or None
        Site identifier extracted from position 2 (zero-indexed),
        or None if filename format is invalid.
    
    Examples
    --------
    >>> extract_site_from_filename("001_42_SiteA_20230615_cam1_clip_003.png")
    'SiteA'
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


def create_dataframe(data_dir, min_dimensions=None):
    """
    Build a DataFrame cataloguing all images in the dataset directory.
    
    Scans the data directory for images organised in class-specific
    subdirectories and extracts metadata including dimensions and site
    information.
    
    Parameters
    ----------
    data_dir : str
        Root directory containing class subdirectories (1_bat, 2_bird,
        3_human, 4_other).
    min_dimensions : int, optional
        Minimum required width and height in pixels. Images smaller than
        this threshold in either dimension are excluded. Default is None
        (no filtering).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - filepath: Absolute path to image file
        - label: Class name string
        - site: Site identifier extracted from filename
        - width: Image width in pixels
        - height: Image height in pixels
        - label_int: Integer class label
    
    Raises
    ------
    ValueError
        If no valid images are found after applying filters.
    
    Notes
    -----
    Uses TensorFlow for image decoding to ensure consistency with the
    training pipeline. Only PNG files are processed.
    """
    print("Initialising dataframe...")
    records = []
    excluded_counts = {label: 0 for label in CLASS_NAMES}
    excluded_counts['total'] = 0

    for label in CLASS_NAMES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Missing class directory: {class_dir} (skipping)")
            continue

        # Filter to PNG files only
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]

        for fname in tqdm(image_files, desc=f"Processing {label}", unit="img"):
            filepath = os.path.join(class_dir, fname)

            # Decode image and extract dimensions using TensorFlow
            try:
                image_bytes = tf.io.read_file(filepath)
                img = tf.io.decode_png(image_bytes, channels=0)
                shape = tf.shape(img)
                height = int(shape[0].numpy())
                width = int(shape[1].numpy())
            except Exception as e:
                tqdm.write(f"[WARNING] Could not read {filepath}: {e} (skipping)")
                continue

            # Apply minimum dimension filter
            if min_dimensions is not None:
                if width < min_dimensions or height < min_dimensions:
                    excluded_counts[label] += 1
                    excluded_counts['total'] += 1
                    continue

            site = extract_site_from_filename(fname)
            records.append({
                'filepath': filepath,
                'label': label,
                'site': site,
                'width': width,
                'height': height
            })

    df = pd.DataFrame(records)
    
    if df.empty:
        raise ValueError(
            f"No valid PNG files found in {data_dir} with class folders "
            f"{CLASS_NAMES} (min_dimensions={min_dimensions})."
        )

    df['label_int'] = df['label'].map(NAME_TO_ID)

    if min_dimensions is not None:
        print(f"\nExcluded {excluded_counts['total']} images below "
              f"{min_dimensions}x{min_dimensions} pixels:")
        for label in CLASS_NAMES:
            print(f"  {label}: {excluded_counts[label]}")

    print(f"\nDataset summary: {len(df)} images across {len(CLASS_NAMES)} classes")
    print(df['label'].value_counts().sort_index())
    
    return df


# =============================================================================
# TENSORFLOW DATA PIPELINE
# =============================================================================

def process_image(filepath, label, target_height, target_width, is_training):
    """
    Load, preprocess, and optionally augment a single image.
    
    Implements a preprocessing pipeline suitable for small infrared
    images from wildlife camera traps. Augmentations are conservative
    to avoid destroying discriminative features in small crops.
    
    Parameters
    ----------
    filepath : tf.Tensor
        String tensor containing the path to the image file.
    label : tf.Tensor
        Integer tensor containing the class label.
    target_height : int
        Desired output height in pixels.
    target_width : int
        Desired output width in pixels.
    is_training : bool
        If True, apply data augmentation; otherwise, only resize.
    
    Returns
    -------
    tuple
        (image, label) where image is a float32 tensor of shape
        (target_height, target_width, 1) with values in [0, 1].
    
    Notes
    -----
    Augmentation strategy for small crops:
    - Random crop (90-100% of original area) then resize back
    - Horizontal flip (vertical preserved for orientation cues)
    - Minor brightness and contrast adjustments
    
    Padding is used during resize to preserve aspect ratio and avoid
    stretching artefacts.
    """
    # Load and decode as grayscale
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if is_training:
        # Store original dimensions for resize-back
        original_shape = tf.shape(image)
        h, w = original_shape[0], original_shape[1]
        
        # Random scale crop: keep 90-100% of area
        scale = tf.random.uniform([], 0.90, 1.0)
        new_h = tf.maximum(1, tf.cast(scale * tf.cast(h, tf.float32), tf.int32))
        new_w = tf.maximum(1, tf.cast(scale * tf.cast(w, tf.float32), tf.int32))
        image = tf.image.random_crop(image, size=[new_h, new_w, original_shape[2]])
        image = tf.image.resize(image, [h, w])
        
        # Geometric augmentation
        image = tf.image.random_flip_left_right(image)
        
        # Photometric augmentation (conservative for IR images)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, 0.95, 1.05)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Resize with padding to preserve aspect ratio
    image = tf.image.resize_with_pad(image, target_height, target_width)
    
    return image, label


def create_tf_dataset(df, target_height, target_width, batch_size, is_training):
    """
    Create a tf.data.Dataset from a DataFrame of image paths and labels.
    
    Implements an efficient data pipeline with parallel loading,
    preprocessing, and prefetching.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'filepath' and 'label_int' columns.
    target_height : int
        Target image height after preprocessing.
    target_width : int
        Target image width after preprocessing.
    batch_size : int
        Number of samples per batch.
    is_training : bool
        If True, shuffle data and apply augmentation.
    
    Returns
    -------
    tf.data.Dataset
        Batched dataset yielding (images, labels) tuples.
    """
    filepaths = df['filepath'].tolist()
    labels = df['label_int'].astype('int32').tolist()

    def _map_fn(fp, lbl):
        return process_image(fp, lbl, target_height, target_width, is_training)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        # Full shuffle with re-shuffling each epoch
        ds = ds.shuffle(
            buffer_size=len(filepaths),
            reshuffle_each_iteration=True
        )

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =============================================================================
# CIFAR-10 PRE-TRAINING UTILITIES
# =============================================================================

def _preprocess_cifar10_image(image, img_height, img_width):
    """
    Convert CIFAR-10 RGB image to grayscale and resize.
    
    Parameters
    ----------
    image : tf.Tensor
        RGB image tensor of shape (32, 32, 3).
    img_height : int
        Target height.
    img_width : int
        Target width.
    
    Returns
    -------
    tf.Tensor
        Grayscale image tensor of shape (img_height, img_width, 1)
        with values normalised to [0, 1].
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def prepare_cifar10_datasets(img_height, img_width, batch_size, val_split=0.1, seed=1234):
    """
    Prepare CIFAR-10 datasets converted to grayscale.
    
    Loads CIFAR-10, converts to grayscale, resizes to target dimensions,
    and creates train/validation/test splits.
    
    Parameters
    ----------
    img_height : int
        Target image height.
    img_width : int
        Target image width.
    batch_size : int
        Batch size for training.
    val_split : float, optional
        Fraction of training data to use for validation.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (train_ds, val_ds, test_ds) tf.data.Dataset objects.
    """
    # Load raw CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar10.load_data()
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    # Shuffle and split training data
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    val_count = int(len(train_images) * val_split)
    val_images, val_labels = train_images[:val_count], train_labels[:val_count]
    train_images, train_labels = train_images[val_count:], train_labels[val_count:]

    def map_fn(image, label):
        return _preprocess_cifar10_image(image, img_height, img_width), label

    # Build datasets with appropriate shuffling
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(len(train_images), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def pretrain_on_cifar10(img_height=64, img_width=64, batch_size=CIFAR_PRETRAIN_BATCH_SIZE,
                        epochs=CIFAR_PRETRAIN_EPOCHS, output_dir="."):
    """
    Pre-train the model backbone on grayscale CIFAR-10.
    
    This transfer learning approach initialises convolutional filters
    with features learned from a larger dataset before fine-tuning on
    the target task. CIFAR-10 is converted to grayscale to match the
    single-channel infrared images.
    
    Parameters
    ----------
    img_height : int
        Target image height (should match fine-tuning).
    img_width : int
        Target image width (should match fine-tuning).
    batch_size : int
        Batch size for pre-training.
    epochs : int
        Maximum pre-training epochs.
    output_dir : str
        Directory to save pre-trained weights.
    
    Returns
    -------
    str
        Path to saved pre-trained weights file.
    
    Notes
    -----
    If weights already exist at the output path, pre-training is skipped
    and the existing path is returned.
    """
    weights_path = os.path.join(output_dir, "cifar10_pretrained_weights.h5")
    
    if os.path.exists(weights_path):
        print(f"Found existing CIFAR-10 weights at {weights_path}; skipping pre-training.")
        return weights_path

    print("Starting CIFAR-10 pre-training (grayscale, resized to target dimensions)...")
    train_ds, val_ds, test_ds = prepare_cifar10_datasets(
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        val_split=CIFAR_PRETRAIN_VAL_SPLIT
    )

    # Build model with 10 classes for CIFAR-10
    model = create_resnet_classifier(
        input_shape=(img_height, img_width, 1),
        num_classes=10,
        learning_rate=CIFAR_PRETRAIN_LR,
        include_bat_metrics=False
    )

    callbacks = [
        EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        LearningRateScheduler(make_lr_schedule(epochs, initial_lr=CIFAR_PRETRAIN_LR))
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Evaluate on test set
    print("\nCIFAR-10 test set evaluation:")
    test_results = model.evaluate(test_ds, verbose=0)
    
    cifar_report = {}
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")
        cifar_report[metric_name] = value
    
    pd.DataFrame([cifar_report]).to_csv(
        os.path.join(output_dir, "cifar_report.csv"), index=False
    )

    model.save_weights(weights_path)
    print(f"Pre-trained weights saved to {weights_path}")

    return weights_path


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def se_block(inputs, reduction=16):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    
    Parameters
    ----------
    inputs : tf.Tensor
        Input feature map of shape (batch, height, width, channels).
    reduction : int, optional
        Reduction ratio for the bottleneck. Default is 16.
    
    Returns
    -------
    tf.Tensor
        Recalibrated feature map with same shape as input.
    
    References
    ----------
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition (pp. 7132-7141).
    """
    filters = inputs.shape[-1]
    
    # Squeeze: global average pooling
    se = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    
    # Excitation: two FC layers with bottleneck
    se = tf.keras.layers.Dense(
        max(filters // reduction, 1),
        activation='relu'
    )(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    
    # Reshape for broadcasting
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    
    # Scale input features
    return tf.keras.layers.Multiply()([inputs, se])


def basic_block(inputs, filters, strides=1, use_se=True, weight_decay=1e-4):
    """
    CIFAR-style basic residual block with SE attention.
    
    Parameters
    ----------
    inputs : tf.Tensor
        Input feature map.
    filters : int
        Number of output filters.
    strides : int, optional
        Stride for the first convolution (used for downsampling).
    use_se : bool, optional
        Whether to include SE attention block.
    weight_decay : float, optional
        L2 regularisation coefficient.
    
    Returns
    -------
    tf.Tensor
        Output feature map with shape determined by filters and strides.
    """
    # First convolution (may downsample)
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=strides, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(weight_decay)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second convolution
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=1, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Optional SE attention
    if use_se:
        x = se_block(x, reduction=16)

    # Shortcut connection with projection if dimensions change
    shortcut = inputs
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=strides, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(weight_decay)
        )(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Residual addition
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def create_resnet_classifier(input_shape=(64, 64, 1), initial_filters=32,
                             block_counts=[2, 2, 2, 2], dropout_rate=0.3,
                             use_se=True, weight_decay=1e-4,
                             learning_rate=FINE_TUNE_LR, num_classes=4,
                             include_bat_metrics=True):
    """
    Build a CIFAR-style ResNet classifier with SE attention.
    
    Creates a compact ResNet architecture optimised for small input
    images typical of object detection crops. Unlike ImageNet-style
    ResNets, this variant:
    - Uses 3x3 convolution in stem (no aggressive initial convolution: 7x7)
    - Omits initial max pooling
    
    Parameters
    ----------
    input_shape : tuple, optional
        Input tensor shape (height, width, channels).
    initial_filters : int, optional
        Number of filters in the stem convolution.
    block_counts : list of int, optional
        Number of residual blocks per stage.
    dropout_rate : float, optional
        Dropout probability before classification head.
    use_se : bool, optional
        Whether to use SE attention in residual blocks.
    weight_decay : float, optional
        L2 regularisation coefficient.
    learning_rate : float, optional
        Initial learning rate for Adam optimiser.
    num_classes : int, optional
        Number of output classes.
    include_bat_metrics : bool, optional
        Whether to include bat-specific PR-AUC metric.
    
    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Stem: single 3x3 conv without max pooling
    x = tf.keras.layers.Conv2D(
        initial_filters, 3, strides=1, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(weight_decay)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Residual stages with progressive downsampling
    filters = initial_filters
    for stage, num_blocks in enumerate(block_counts):
        for block in range(num_blocks):
            # Downsample at first block of stages 1, 2, 3 (not stage 0)
            stride = 2 if (stage > 0 and block == 0) else 1
            x = basic_block(x, filters, strides=stride, use_se=use_se,
                           weight_decay=weight_decay)
        filters *= 2  # Double filters after each stage

    # Global pooling and classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='class')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=get_training_metrics(include_bat_metrics)
    )
    
    return model


# =============================================================================
# CUSTOM METRICS
# =============================================================================

class BatPRAUC(tf.keras.metrics.AUC):
    """
    Precision-Recall Area Under Curve for the bat class.
    
    Monitors PR-AUC specifically for the bat class during training,
    used for early stopping and model selection where bat detection
    is the primary objective of the classifier.
    
    Parameters
    ----------
    bat_id : int, optional
        Integer label for the bat class.
    name : str, optional
        Name of the metric.
    """
    
    def __init__(self, bat_id=0, name='bat_pr_auc', **kwargs):
        super().__init__(curve='PR', name=name, **kwargs)
        self.bat_id = bat_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert multiclass labels to binary bat/not-bat
        y_true_bat = tf.cast(tf.equal(y_true, self.bat_id), self.dtype)
        # Extract bat class probability
        p_bat = y_pred[..., self.bat_id]
        return super().update_state(y_true_bat, p_bat, sample_weight)


def get_training_metrics(include_bat_metrics=True):
    """
    Get list of metrics for model compilation.
    
    Parameters
    ----------
    include_bat_metrics : bool
        Whether to include bat-specific PR-AUC metric.
    
    Returns
    -------
    list
        List of metric objects/names.
    """
    base_metrics = [
        'sparse_categorical_accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')
    ]
    
    if not include_bat_metrics:
        return base_metrics
    
    return base_metrics + [BatPRAUC(bat_id=BAT_ID, name='bat_pr_auc')]


# =============================================================================
# LEARNING RATE SCHEDULING
# =============================================================================

def make_lr_schedule(total_epochs, warmup_epochs=None, initial_lr=FINE_TUNE_LR,
                     min_lr=1e-5):
    """
    Create a cosine decay learning rate schedule with linear warmup.
    
    Parameters
    ----------
    total_epochs : int
        Total number of training epochs.
    warmup_epochs : int, optional
        Number of warmup epochs. Defaults to 10% of total epochs.
    initial_lr : float
        Peak learning rate after warmup.
    min_lr : float
        Minimum learning rate at end of decay.
    
    Returns
    -------
    callable
        Learning rate schedule function for LearningRateScheduler callback.
    
    References
    ----------
    Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
    with Warm Restarts. In Proceedings of ICLR.
    """
    if warmup_epochs is None:
        warmup_epochs = max(1, int(0.1 * total_epochs))

    def schedule(epoch, lr):
        # Linear warmup
        if epoch < warmup_epochs:
            return initial_lr * float(epoch + 1) / float(warmup_epochs)
        
        if total_epochs <= warmup_epochs:
            return float(initial_lr)
        
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        progress = np.clip(progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        decayed_lr = (initial_lr - min_lr) * cosine_decay + min_lr
        
        return float(decayed_lr)

    return schedule


# =============================================================================
# CLASS WEIGHTING
# =============================================================================

def calculate_class_weights(df, bat_multiplier=2.0):
    """
    Calculate balanced class weights with enhanced bat emphasis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'label_int' column.
    bat_multiplier : float
        Additional weight multiplier for bat class.
    
    Returns
    -------
    dict
        Mapping from class index to weight value.
    
    Notes
    -----
    Weights are normalised so their mean equals 1.0
    """
    y = df['label_int'].values
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    
    # Extra emphasis on bat class
    class_weights[BAT_ID] *= bat_multiplier
    
    # Normalise to mean of 1
    class_weights = class_weights / class_weights.mean()
    
    return dict(enumerate(class_weights))


# =============================================================================
# TRANSFER LEARNING UTILITIES
# =============================================================================

def load_pretrained_backbone_weights(model, weights_path):
    """
    Load pre-trained weights into model backbone, skipping head layers.
    
    Parameters
    ----------
    model : tf.keras.Model
        Target model to load weights into.
    weights_path : str or None
        Path to weights file, or None to skip loading.
    
    Returns
    -------
    tf.keras.Model
        Model with loaded weights (modified in place).
    """
    if not weights_path:
        return model
    
    if not os.path.exists(weights_path):
        print(f"[WARNING] Pre-trained weights not found at {weights_path}")
        return model

    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"Loaded pre-trained backbone weights from {weights_path}")
    except Exception as exc:
        print(f"[WARNING] Could not load weights from {weights_path}: {exc}")
    
    return model


def run_head_start_phase(model, train_ds, val_ds, class_weights, fine_tune_lr,
                         epochs=HEAD_START_EPOCHS):
    """
    Train only the classification head before full fine-tuning.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model with pre-trained backbone weights.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset or None
        Validation dataset (optional).
    class_weights : dict
        Class weight dictionary.
    fine_tune_lr : float
        Learning rate for subsequent fine-tuning phase.
    epochs : int
        Number of head-only training epochs. Set to 0 to skip.
    
    Notes
    -----
    The model is modified in place. After head-start training, all
    layers are unfrozen and the model is recompiled with fine_tune_lr.
    """
    if epochs <= 0:
        return

    print(f"\n--- Head-start phase: {epochs} epochs (LR={HEAD_START_LR}) ---")
    
    # Freeze all layers except classification head
    for layer in model.layers:
        if layer.name != 'class':
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=HEAD_START_LR),
        loss='sparse_categorical_crossentropy',
        metrics=get_training_metrics(include_bat_metrics=True)
    )

    fit_kwargs = {"epochs": epochs, "class_weight": class_weights}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds

    model.fit(train_ds, **fit_kwargs)

    # Unfreeze all layers for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=get_training_metrics(include_bat_metrics=True)
    )
    
    print("--- Head-start phase complete; all layers unfrozen ---\n")


# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

def pick_optimal_threshold(y_true_int, p_bat, strategy='fbeta', beta=1.0,
                          target_precision=None, target_recall=None,
                          bat_id=BAT_ID):
    """
    Select optimal decision threshold for bat classification.
    
    Offers multiple strategies for threshold selection depending on
    the application requirements (e.g., maximising F-beta, achieving
    minimum precision, or achieving minimum recall).
    
    Parameters
    ----------
    y_true_int : np.ndarray
        True class labels (integer encoded).
    p_bat : np.ndarray
        Predicted bat probabilities.
    strategy : str
        Threshold selection strategy:
        - 'fbeta': Maximise F-beta score
        - 'precision_constrained': Maximise recall at minimum precision
        - 'recall_constrained': Maximise precision at minimum recall
    beta : float
        Beta parameter for F-beta score. Values > 1 favour recall;
        values < 1 favour precision.
    target_precision : float, optional
        Minimum acceptable precision for 'precision_constrained'.
    target_recall : float, optional
        Minimum acceptable recall for 'recall_constrained'.
    bat_id : int
        Integer label for bat class.
    
    Returns
    -------
    float
        Selected probability threshold in [0, 1].
    
    Notes
    -----
    If the constrained strategies cannot meet their targets, the
    function falls back to F1 optimisation (beta=1.0) with a warning.
    """
    # Convert to binary labels
    y_true_bin = (y_true_int == bat_id).astype(int)
    
    if y_true_bin.sum() == 0:
        print("[WARNING] No bat samples for threshold calibration; using 0.5")
        return 0.5
    
    prec, rec, thr = precision_recall_curve(y_true_bin, p_bat)
    
    if len(thr) == 0:
        return 0.5
    
    # Remove edge cases where precision or recall is zero
    valid_idx = (prec[:-1] > 0) & (rec[:-1] > 0)
    
    # Precision-constrained: maximise recall at minimum precision
    if strategy == 'precision_constrained' and target_precision is not None:
        valid_idx_constrained = valid_idx & (prec[:-1] >= target_precision)
        if not valid_idx_constrained.any():
            print(f"[WARNING] Cannot achieve precision >= {target_precision:.2f}; "
                  f"falling back to F1 optimisation")
            strategy = 'fbeta'
            beta = 1.0  # Reset to F1
        else:
            best_idx = np.where(valid_idx_constrained)[0][
                np.argmax(rec[:-1][valid_idx_constrained])
            ]
            threshold = float(thr[best_idx])
            print(f"Precision-constrained threshold {threshold:.3f}: "
                  f"P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}")
            return threshold
    
    # Recall-constrained: maximise precision at minimum recall
    if strategy == 'recall_constrained' and target_recall is not None:
        valid_idx_constrained = valid_idx & (rec[:-1] >= target_recall)
        if not valid_idx_constrained.any():
            print(f"[WARNING] Cannot achieve recall >= {target_recall:.2f}; "
                  f"falling back to F1 optimisation")
            strategy = 'fbeta'
            beta = 1.0  # Reset to F1
        else:
            best_idx = np.where(valid_idx_constrained)[0][
                np.argmax(prec[:-1][valid_idx_constrained])
            ]
            threshold = float(thr[best_idx])
            print(f"Recall-constrained threshold {threshold:.3f}: "
                  f"P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}")
            return threshold
    
    # F-beta optimisation (default or fallback)
    if not valid_idx.any():
        return 0.5
    
    beta2 = beta ** 2
    fbeta = (1 + beta2) * (prec[:-1] * rec[:-1]) / (beta2 * prec[:-1] + rec[:-1] + 1e-8)
    fbeta[~valid_idx] = 0
    
    best_idx = np.argmax(fbeta)
    threshold = float(thr[best_idx])
    
    print(f"F{beta:.1f}-optimal threshold: {threshold:.3f} -> "
          f"P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}, F{beta:.1f}={fbeta[best_idx]:.3f}")
    
    return threshold


# =============================================================================
# EVALUATION AND VISUALISATION
# =============================================================================

def compute_multiclass_curves(y_true_int, y_prob, present_labels):
    """
    Compute ROC and PR curves for multiclass classification.
    
    Calculates per-class and micro/macro-averaged curves with
    corresponding AUC and AP scores.
    
    Parameters
    ----------
    y_true_int : np.ndarray
        True class labels (integer encoded).
    y_prob : np.ndarray
        Predicted class probabilities, shape (n_samples, n_classes).
    present_labels : array-like
        List of class labels present in y_true_int.
    
    Returns
    -------
    curves_roc : dict
        ROC curves indexed by class label and 'micro'.
    curves_pr : dict
        PR curves indexed by class label and 'micro'.
    aucs : dict
        ROC-AUC scores indexed by class label, 'micro', and 'macro'.
    aps : dict
        Average precision scores indexed by class label, 'micro', and 'macro'.
    """
    y_true_bin = label_binarize(y_true_int, classes=CLASS_LABELS)
    curves_roc, curves_pr, aucs, aps = {}, {}, {}, {}

    # Micro-averaged curves (across all classes)
    valid_cols = [c for c in present_labels if y_true_bin[:, c].sum() > 0]
    
    if valid_cols:
        y_true_micro = y_true_bin[:, valid_cols].ravel()
        y_prob_micro = y_prob[:, valid_cols].ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_true_micro, y_prob_micro)
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true_micro, y_prob_micro
        )
        curves_roc['micro'] = (fpr_micro, tpr_micro)
        curves_pr['micro'] = (recall_micro, precision_micro)
        aucs['micro'] = auc(fpr_micro, tpr_micro)
        aps['micro'] = average_precision_score(y_true_micro, y_prob_micro)
    else:
        aucs['micro'] = np.nan
        aps['micro'] = np.nan

    # Per-class curves
    for c in CLASS_LABELS:
        if c not in present_labels:
            curves_roc[c] = None
            curves_pr[c] = None
            aucs[c] = np.nan
            aps[c] = np.nan
            continue
        
        y_true_c = y_true_bin[:, c]
        if y_true_c.sum() == 0:
            curves_roc[c] = None
            curves_pr[c] = None
            aucs[c] = np.nan
            aps[c] = np.nan
            continue
            
        fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
        precision, recall, _ = precision_recall_curve(y_true_c, y_prob[:, c])
        curves_roc[c] = (fpr, tpr)
        curves_pr[c] = (recall, precision)
        aucs[c] = auc(fpr, tpr)
        aps[c] = average_precision_score(y_true_c, y_prob[:, c])

    # Macro-averaged scores
    valid_auc_vals = [aucs[c] for c in CLASS_LABELS if not np.isnan(aucs[c])]
    valid_ap_vals = [aps[c] for c in CLASS_LABELS if not np.isnan(aps[c])]
    aucs['macro'] = np.mean(valid_auc_vals) if valid_auc_vals else np.nan
    aps['macro'] = np.mean(valid_ap_vals) if valid_ap_vals else np.nan
    
    return curves_roc, curves_pr, aucs, aps


def plot_threshold_analysis(y_true_int, p_bat, selected_threshold, output_path,
                           bat_id=BAT_ID, beta=2.0):
    """
    Generate diagnostic plots for threshold selection.
    
    Creates a 2x2 figure showing:
    - Probability distributions by true class
    - Precision-recall curve with operating point
    - F1 and F-beta scores vs threshold
    - Precision and recall vs threshold
    
    Parameters
    ----------
    y_true_int : np.ndarray
        True class labels.
    p_bat : np.ndarray
        Predicted bat probabilities.
    selected_threshold : float
        The chosen threshold to highlight.
    output_path : str
        Path to save the figure.
    bat_id : int
        Integer label for bat class.
    beta : float
        Beta value for F-beta score visualisation.
    """
    y_true_bin = (y_true_int == bat_id).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Probability distributions
    ax = axes[0, 0]
    ax.hist(p_bat[y_true_bin == 0], bins=50, alpha=0.5, label='Non-bat',
            color='blue', density=True)
    ax.hist(p_bat[y_true_bin == 1], bins=50, alpha=0.5, label='Bat',
            color='red', density=True)
    ax.axvline(selected_threshold, color='black', linestyle='--',
               label=f'Threshold: {selected_threshold:.3f}')
    ax.set_xlabel('P(bat)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Bat Probabilities')
    ax.legend()
    
    # Panel 2: Precision-recall curve
    ax = axes[0, 1]
    prec, rec, thr = precision_recall_curve(y_true_bin, p_bat)
    ax.plot(rec, prec, 'b-', label='PR curve')
    
    if len(thr) > 0:
        idx = np.argmin(np.abs(thr - selected_threshold))
        ax.plot(rec[idx], prec[idx], 'ro', markersize=10,
                label=f'Operating point (P={prec[idx]:.2f}, R={rec[idx]:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: F-scores vs threshold
    ax = axes[1, 0]
    if len(thr) > 0:
        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
        beta2 = beta ** 2
        fbeta_scores = (1 + beta2) * (prec[:-1] * rec[:-1]) / \
                       (beta2 * prec[:-1] + rec[:-1] + 1e-8)
        
        ax.plot(thr, f1_scores, 'g-', label='F1')
        ax.plot(thr, fbeta_scores, 'm-', label=f'F{beta:.1f}')
        ax.axvline(selected_threshold, color='black', linestyle='--',
                   label='Selected threshold')
        
        idx_thr = np.argmin(np.abs(thr - selected_threshold))
        ax.plot(thr[idx_thr], fbeta_scores[idx_thr], 'ko', markersize=6)
        
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'F1 and F{beta:.1f} vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Precision and recall vs threshold
    ax = axes[1, 1]
    if len(thr) > 0:
        ax.plot(thr, prec[:-1], 'b-', label='Precision')
        ax.plot(thr, rec[:-1], 'r-', label='Recall')
        ax.axvline(selected_threshold, color='black', linestyle='--',
                   label='Selected threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualise_random_transformations(row, target_height, target_width,
                                     num_rows=3, num_cols=3, output_dir=None):
    """
    Display a grid of augmented versions of a single image.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing 'filepath', 'label', and 'label_int'.
    target_height : int
        Target image height.
    target_width : int
        Target image width.
    num_rows : int
        Number of rows in the grid.
    num_cols : int
        Number of columns in the grid.
    """
    sample_filepath = row['filepath']
    label_text = row['label']

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_rows * num_cols):
        aug_image, _ = process_image(
            sample_filepath, row['label_int'],
            target_height, target_width, is_training=True
        )
        aug_image_np = tf.clip_by_value(aug_image, 0, 1).numpy()
        aug_image_np = np.squeeze(aug_image_np, axis=-1)
        axes[i].imshow(aug_image_np, cmap='gray')
        axes[i].axis("off")
        axes[i].set_title(f"Aug {i+1}\nLabel: {label_text}")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"augmentation_preview_{label_text}.png"), dpi=150)
    else:
        plt.show()
    plt.close()


# =============================================================================
# CONFIGURATION LOADING AND PARSING
# =============================================================================

def load_config_from_ini(config_path):
    """
    Load configuration from .ini file.

    Parameters
    ----------
    config_path : str
        Path to .ini configuration file.

    Returns
    -------
    dict
        Dictionary containing all configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    # Define default values matching original hardcoded configuration
    defaults = {
        # Paths
        'data_dir': 'path/to/data',
        'output_dir': 'path/to/output',

        # Image parameters
        'min_dimensions': 10,
        'img_height': 64,
        'img_width': 64,
        'channels': 1,

        # Training hyperparameters
        'batch_size': 64,
        'epochs': 25,
        'k_folds': 16,

        # Class weighting
        'use_class_weights': True,
        'bat_weight_multiplier': 2.0,

        # CIFAR-10 pre-training
        'pretrain_on_cifar10': True,
        'cifar_pretrain_epochs': 15,
        'cifar_pretrain_batch_size': 256,
        'cifar_pretrain_val_split': 0.1,
        'cifar_pretrain_lr': 1e-3,

        # Fine-tuning parameters
        'fine_tune_lr': 1e-4,
        'head_start_epochs': 5,
        'head_start_lr': 3e-4,

        # Model architecture
        'initial_filters': 32,
        'dropout_rate': 0.3,
        'use_se': True,
        'weight_decay': 1e-4,

        # Threshold calibration
        'threshold_strategy': 'fbeta',
        'threshold_beta': 2.0,
        'target_precision': 0.8,
        'target_recall': 0.9,

        # Pipeline control
        'visualise_transformations': False,
        'k_fold_val': True,
        'train_production_model': True,

        # Data split
        'holdout_fraction': 0.2,
        'random_seed': 1234,
    }

    # Parse configuration sections
    cfg = {}

    # Paths section
    if config.has_section('paths'):
        cfg['data_dir'] = config.get('paths', 'data_dir', fallback=defaults['data_dir'])
        cfg['output_dir'] = config.get('paths', 'output_dir', fallback=defaults['output_dir'])
    else:
        cfg['data_dir'] = defaults['data_dir']
        cfg['output_dir'] = defaults['output_dir']

    # Image parameters section
    if config.has_section('image'):
        cfg['min_dimensions'] = config.getint('image', 'min_dimensions', fallback=defaults['min_dimensions'])
        cfg['img_height'] = config.getint('image', 'img_height', fallback=defaults['img_height'])
        cfg['img_width'] = config.getint('image', 'img_width', fallback=defaults['img_width'])
        cfg['channels'] = config.getint('image', 'channels', fallback=defaults['channels'])
    else:
        cfg['min_dimensions'] = defaults['min_dimensions']
        cfg['img_height'] = defaults['img_height']
        cfg['img_width'] = defaults['img_width']
        cfg['channels'] = defaults['channels']

    # Training section
    if config.has_section('training'):
        cfg['batch_size'] = config.getint('training', 'batch_size', fallback=defaults['batch_size'])
        cfg['epochs'] = config.getint('training', 'epochs', fallback=defaults['epochs'])
        cfg['k_folds'] = config.getint('training', 'k_folds', fallback=defaults['k_folds'])
        cfg['use_class_weights'] = config.getboolean('training', 'use_class_weights', fallback=defaults['use_class_weights'])
        cfg['bat_weight_multiplier'] = config.getfloat('training', 'bat_weight_multiplier', fallback=defaults['bat_weight_multiplier'])
    else:
        cfg['batch_size'] = defaults['batch_size']
        cfg['epochs'] = defaults['epochs']
        cfg['k_folds'] = defaults['k_folds']
        cfg['use_class_weights'] = defaults['use_class_weights']
        cfg['bat_weight_multiplier'] = defaults['bat_weight_multiplier']

    # CIFAR pre-training section
    if config.has_section('cifar_pretrain'):
        cfg['pretrain_on_cifar10'] = config.getboolean('cifar_pretrain', 'enabled', fallback=defaults['pretrain_on_cifar10'])
        cfg['cifar_pretrain_epochs'] = config.getint('cifar_pretrain', 'epochs', fallback=defaults['cifar_pretrain_epochs'])
        cfg['cifar_pretrain_batch_size'] = config.getint('cifar_pretrain', 'batch_size', fallback=defaults['cifar_pretrain_batch_size'])
        cfg['cifar_pretrain_val_split'] = config.getfloat('cifar_pretrain', 'val_split', fallback=defaults['cifar_pretrain_val_split'])
        cfg['cifar_pretrain_lr'] = config.getfloat('cifar_pretrain', 'learning_rate', fallback=defaults['cifar_pretrain_lr'])
    else:
        cfg['pretrain_on_cifar10'] = defaults['pretrain_on_cifar10']
        cfg['cifar_pretrain_epochs'] = defaults['cifar_pretrain_epochs']
        cfg['cifar_pretrain_batch_size'] = defaults['cifar_pretrain_batch_size']
        cfg['cifar_pretrain_val_split'] = defaults['cifar_pretrain_val_split']
        cfg['cifar_pretrain_lr'] = defaults['cifar_pretrain_lr']

    # Fine-tuning section
    if config.has_section('fine_tuning'):
        cfg['fine_tune_lr'] = config.getfloat('fine_tuning', 'learning_rate', fallback=defaults['fine_tune_lr'])
        cfg['head_start_epochs'] = config.getint('fine_tuning', 'head_start_epochs', fallback=defaults['head_start_epochs'])
        cfg['head_start_lr'] = config.getfloat('fine_tuning', 'head_start_lr', fallback=defaults['head_start_lr'])
    else:
        cfg['fine_tune_lr'] = defaults['fine_tune_lr']
        cfg['head_start_epochs'] = defaults['head_start_epochs']
        cfg['head_start_lr'] = defaults['head_start_lr']

    # Model architecture section
    if config.has_section('model'):
        cfg['initial_filters'] = config.getint('model', 'initial_filters', fallback=defaults['initial_filters'])
        cfg['dropout_rate'] = config.getfloat('model', 'dropout_rate', fallback=defaults['dropout_rate'])
        cfg['use_se'] = config.getboolean('model', 'use_se', fallback=defaults['use_se'])
        cfg['weight_decay'] = config.getfloat('model', 'weight_decay', fallback=defaults['weight_decay'])
    else:
        cfg['initial_filters'] = defaults['initial_filters']
        cfg['dropout_rate'] = defaults['dropout_rate']
        cfg['use_se'] = defaults['use_se']
        cfg['weight_decay'] = defaults['weight_decay']

    # Threshold calibration section
    if config.has_section('threshold'):
        cfg['threshold_strategy'] = config.get('threshold', 'strategy', fallback=defaults['threshold_strategy'])
        cfg['threshold_beta'] = config.getfloat('threshold', 'beta', fallback=defaults['threshold_beta'])
        cfg['target_precision'] = config.getfloat('threshold', 'target_precision', fallback=defaults['target_precision'])
        cfg['target_recall'] = config.getfloat('threshold', 'target_recall', fallback=defaults['target_recall'])
    else:
        cfg['threshold_strategy'] = defaults['threshold_strategy']
        cfg['threshold_beta'] = defaults['threshold_beta']
        cfg['target_precision'] = defaults['target_precision']
        cfg['target_recall'] = defaults['target_recall']

    # Pipeline section
    if config.has_section('pipeline'):
        cfg['visualise_transformations'] = config.getboolean('pipeline', 'visualise_transformations', fallback=defaults['visualise_transformations'])
        cfg['k_fold_val'] = config.getboolean('pipeline', 'k_fold_val', fallback=defaults['k_fold_val'])
        cfg['train_production_model'] = config.getboolean('pipeline', 'train_production_model', fallback=defaults['train_production_model'])
        cfg['holdout_fraction'] = config.getfloat('pipeline', 'holdout_fraction', fallback=defaults['holdout_fraction'])
        cfg['random_seed'] = config.getint('pipeline', 'random_seed', fallback=defaults['random_seed'])
    else:
        cfg['visualise_transformations'] = defaults['visualise_transformations']
        cfg['k_fold_val'] = defaults['k_fold_val']
        cfg['train_production_model'] = defaults['train_production_model']
        cfg['holdout_fraction'] = defaults['holdout_fraction']
        cfg['random_seed'] = defaults['random_seed']

    return cfg


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Multi-class Image Classification Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using config file only
  python classification_training.py --config config.ini

  # Using CLI arguments only
  python classification_training.py --data_dir /data --output_dir /output --epochs 30

  # Mixed (CLI overrides config file)
  python classification_training.py --config config.ini --epochs 50 --batch_size 128
        """
    )

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to .ini configuration file')

    # Paths
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to data directory containing class subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to output directory for results')

    # Image parameters
    parser.add_argument('--min_dimensions', type=int, default=None,
                       help='Minimum image dimensions (width and height)')
    parser.add_argument('--img_height', type=int, default=None,
                       help='Target image height for training')
    parser.add_argument('--img_width', type=int, default=None,
                       help='Target image width for training')
    parser.add_argument('--channels', type=int, default=None,
                       help='Number of image channels (1 for grayscale)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--k_folds', type=int, default=None,
                       help='Number of folds for cross-validation')

    # Class weighting
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to use class weights (true/false)')
    parser.add_argument('--bat_weight_multiplier', type=float, default=None,
                       help='Additional weight multiplier for bat class')

    # CIFAR-10 pre-training
    parser.add_argument('--pretrain_on_cifar10', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to pre-train on CIFAR-10 (true/false)')
    parser.add_argument('--cifar_pretrain_epochs', type=int, default=None,
                       help='Number of CIFAR-10 pre-training epochs')
    parser.add_argument('--cifar_pretrain_batch_size', type=int, default=None,
                       help='Batch size for CIFAR-10 pre-training')
    parser.add_argument('--cifar_pretrain_val_split', type=float, default=None,
                       help='Validation split for CIFAR-10 pre-training')
    parser.add_argument('--cifar_pretrain_lr', type=float, default=None,
                       help='Learning rate for CIFAR-10 pre-training')

    # Fine-tuning
    parser.add_argument('--fine_tune_lr', type=float, default=None,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--head_start_epochs', type=int, default=None,
                       help='Number of head-only training epochs')
    parser.add_argument('--head_start_lr', type=float, default=None,
                       help='Learning rate for head-only training')

    # Model architecture
    parser.add_argument('--initial_filters', type=int, default=None,
                       help='Number of initial filters in ResNet')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Dropout rate before classification head')
    parser.add_argument('--use_se', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to use SE blocks (true/false)')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='L2 weight decay regularization')

    # Threshold calibration
    parser.add_argument('--threshold_strategy', type=str, default=None,
                       choices=['fbeta', 'precision_constrained', 'recall_constrained'],
                       help='Threshold selection strategy')
    parser.add_argument('--threshold_beta', type=float, default=None,
                       help='Beta value for F-beta threshold optimization')
    parser.add_argument('--target_precision', type=float, default=None,
                       help='Target precision for precision_constrained strategy')
    parser.add_argument('--target_recall', type=float, default=None,
                       help='Target recall for recall_constrained strategy')

    # Pipeline control
    parser.add_argument('--visualise_transformations', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to visualize data augmentations (true/false)')
    parser.add_argument('--k_fold_val', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to perform k-fold cross-validation (true/false)')
    parser.add_argument('--train_production_model', type=lambda x: x.lower() == 'true', default=None,
                       help='Whether to train production model on all data (true/false)')

    # Data split
    parser.add_argument('--holdout_fraction', type=float, default=None,
                       help='Fraction of sites to reserve for holdout evaluation')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility')

    return parser.parse_args()


def merge_config(config_dict, args):
    """
    Merge configuration from file and command-line arguments.

    Command-line arguments take precedence over config file values.

    Parameters
    ----------
    config_dict : dict
        Configuration loaded from .ini file.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    # Start with config file values (or defaults if no config file)
    cfg = config_dict.copy()

    # Override with CLI arguments where provided
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            cfg[key] = value

    return cfg


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df_train, output_path, IMG_HEIGHT=64, IMG_WIDTH=64,
                      channels=1, batch_size=64, epochs=35,
                      pretrained_weights=None):
    """
    Train the final model on all available training data.
    
    Unlike cross-validation training, this function trains a single model
    on the complete training set.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data DataFrame.
    output_path : str
        Path to save the trained model.
    IMG_HEIGHT : int
        Input image height.
    IMG_WIDTH : int
        Input image width.
    channels : int
        Number of input channels (1 for grayscale).
    batch_size : int
        Training batch size.
    epochs : int
        Number of training epochs.
    pretrained_weights : str, optional
        Path to pre-trained backbone weights.
    
    Returns
    -------
    tf.keras.Model
        Trained model.
    """
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    
    print(f"\nTraining sites: {sorted(df_train['site'].dropna().unique())}")
    print(f"\nLabel distribution:\n{df_train['label_int'].value_counts().sort_index()}")

    # Calculate class weights
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(df_train, BAT_WEIGHT_MULTIPLIER)
        print(f"\nClass weights: {{{', '.join(f'{SHORT_NAMES[k]}: {v:.2f}' for k, v in class_weights.items())}}}")
    else:
        class_weights = None

    train_ds = create_tf_dataset(df_train, IMG_HEIGHT, IMG_WIDTH, batch_size,
                                 is_training=True)

    # Build and initialise model
    model = create_resnet_classifier(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, channels),
        num_classes=len(CLASS_LABELS),
        learning_rate=FINE_TUNE_LR
    )
    model = load_pretrained_backbone_weights(model, pretrained_weights)

    # Optional head-only warmup
    run_head_start_phase(
        model, train_ds, None, class_weights,
        fine_tune_lr=FINE_TUNE_LR, epochs=HEAD_START_EPOCHS
    )
    
    model.summary()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        LearningRateScheduler(make_lr_schedule(epochs, initial_lr=FINE_TUNE_LR))
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    model.save(output_path)
    print(f"\nModel saved to {output_path}")
    
    return model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the classification training pipeline.

    Orchestrates the complete workflow:
    1. Configuration loading from .ini file and/or CLI arguments
    2. Data loading and site-based splitting
    3. CIFAR-10 pre-training
    4. Stratified Group K-Fold cross-validation
    5. Threshold calibration on pooled out-of-fold predictions
    6. Final model training for holdout evaluation
    7. Optional production model training on all data
    """
    # -------------------------------------------------------------------------
    # CONFIGURATION LOADING
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration from file if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"ERROR: Configuration file not found: {args.config}")
            sys.exit(1)
        print(f"Loading configuration from: {args.config}")
        config_dict = load_config_from_ini(args.config)
    else:
        # Use defaults if no config file provided
        print("No configuration file provided. Using defaults and CLI arguments.")
        config_dict = load_config_from_ini(None)  # Will use all defaults

    # Merge config file with CLI arguments (CLI takes precedence)
    cfg = merge_config(config_dict, args)

    # Update global constants that are used by other functions
    global USE_CLASS_WEIGHTS, BAT_WEIGHT_MULTIPLIER, PRETRAIN_ON_CIFAR10
    global CIFAR_PRETRAIN_EPOCHS, CIFAR_PRETRAIN_BATCH_SIZE, CIFAR_PRETRAIN_VAL_SPLIT, CIFAR_PRETRAIN_LR
    global FINE_TUNE_LR, HEAD_START_EPOCHS, HEAD_START_LR

    USE_CLASS_WEIGHTS = cfg['use_class_weights']
    BAT_WEIGHT_MULTIPLIER = cfg['bat_weight_multiplier']
    PRETRAIN_ON_CIFAR10 = cfg['pretrain_on_cifar10']
    CIFAR_PRETRAIN_EPOCHS = cfg['cifar_pretrain_epochs']
    CIFAR_PRETRAIN_BATCH_SIZE = cfg['cifar_pretrain_batch_size']
    CIFAR_PRETRAIN_VAL_SPLIT = cfg['cifar_pretrain_val_split']
    CIFAR_PRETRAIN_LR = cfg['cifar_pretrain_lr']
    FINE_TUNE_LR = cfg['fine_tune_lr']
    HEAD_START_EPOCHS = cfg['head_start_epochs']
    HEAD_START_LR = cfg['head_start_lr']

    # Extract configuration values for local use
    data_dir = cfg['data_dir']
    output_dir = cfg['output_dir']
    MIN_DIMENSIONS = cfg['min_dimensions']
    IMG_HEIGHT = cfg['img_height']
    IMG_WIDTH = cfg['img_width']
    channels = cfg['channels']
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    k_folds = cfg['k_folds']
    THRESHOLD_STRATEGY = cfg['threshold_strategy']
    THRESHOLD_BETA = cfg['threshold_beta']
    TARGET_PRECISION = cfg['target_precision']
    TARGET_RECALL = cfg['target_recall']
    VISUALISE_TRANSFORMATIONS = cfg['visualise_transformations']
    K_FOLD_VAL = cfg['k_fold_val']
    TRAIN_PRODUCTION_MODEL = cfg['train_production_model']

    # Print configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}, channels: {channels}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, K-folds: {k_folds}")
    print(f"Use class weights: {USE_CLASS_WEIGHTS}, Bat weight multiplier: {BAT_WEIGHT_MULTIPLIER}")
    print(f"CIFAR-10 pre-training: {PRETRAIN_ON_CIFAR10}")
    print(f"Threshold strategy: {THRESHOLD_STRATEGY}, Beta: {THRESHOLD_BETA}")
    print(f"K-fold validation: {K_FOLD_VAL}, Production model: {TRAIN_PRODUCTION_MODEL}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    df = create_dataframe(data_dir, min_dimensions=MIN_DIMENSIONS)

    # Optional: visualise augmentations
    if VISUALISE_TRANSFORMATIONS:
        samples = df.groupby('label', group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), 5), random_state=cfg['random_seed'])
        )
        for _, sample_row in samples.iterrows():
            visualise_random_transformations(sample_row, IMG_HEIGHT, IMG_WIDTH, output_dir=output_dir)

    # Site-based train/holdout split
    unique_sites = np.array(sorted([s for s in df['site'].dropna().unique()]))
    holdout_size = max(1, int(cfg['holdout_fraction'] * len(unique_sites))) if len(unique_sites) > 0 else 0

    np.random.seed(cfg['random_seed'])
    holdout_sites = np.random.choice(unique_sites, size=holdout_size, replace=False) \
                    if holdout_size > 0 else np.array([])
    train_sites = np.setdiff1d(unique_sites, holdout_sites)

    df_train = df[df['site'].isin(train_sites)].copy() if len(train_sites) else df.iloc[[]].copy()
    df_holdout = df[df['site'].isin(holdout_sites)].copy() if len(holdout_sites) else df.iloc[[]].copy()

    print(f"\nSite allocation:")
    print(f"  All sites ({len(unique_sites)}): {list(unique_sites)}")
    print(f"  Training sites ({len(train_sites)}): {list(train_sites)}")
    print(f"  Holdout sites ({len(holdout_sites)}): {list(holdout_sites)}")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # OPTIONAL: CIFAR-10 PRE-TRAINING
    # -------------------------------------------------------------------------
    pretrained_weights_path = None
    if PRETRAIN_ON_CIFAR10:
        pretrained_weights_path = pretrain_on_cifar10(
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            batch_size=CIFAR_PRETRAIN_BATCH_SIZE,
            epochs=CIFAR_PRETRAIN_EPOCHS,
            output_dir=output_dir
        )
    
    # Save data split information for reproducibility
    split_log = pd.DataFrame({
        "split": ["training", "holdout"],
        "sites": [", ".join(train_sites), ", ".join(holdout_sites)]
    })
    split_log.to_csv(os.path.join(output_dir, "data_split_log.csv"), index=False)

    # Save class weights for reference
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(df_train, BAT_WEIGHT_MULTIPLIER)
        pd.DataFrame([{SHORT_NAMES[k]: v for k, v in class_weights.items()}]).to_csv(
            os.path.join(output_dir, "class_weights.csv"), index=False
        )

    # -------------------------------------------------------------------------
    # K-FOLD CROSS-VALIDATION
    # -------------------------------------------------------------------------
    # Tracking variables for aggregated results
    fold_metrics = []
    best_epochs = []
    oof_y_true = []
    oof_y_prob = []
    fold_class_stats = []
    fold_avg_stats = []

    if K_FOLD_VAL and len(train_sites) >= k_folds:
        print("\n" + "=" * 60)
        print(f"STRATIFIED GROUP K-FOLD CROSS-VALIDATION (k={k_folds})")
        print("=" * 60)
        
        groups = df_train['site']
        labels = df_train['label']
        
        # Stratified Group K-Fold ensures:
        # - No site appears in both train and validation
        # - Class proportions are maintained in each fold
        cv = StratifiedGroupKFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=cfg['random_seed']
        )
        
        for fold_num, (train_index, valid_index) in enumerate(
            cv.split(df_train, y=labels, groups=groups), start=1
        ):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_num}/{k_folds}")
            print(f"{'='*60}")
            
            train_df = df_train.iloc[train_index].reset_index(drop=True)
            valid_df = df_train.iloc[valid_index].reset_index(drop=True)
            
            train_sites_fold = sorted(train_df['site'].dropna().unique())
            valid_sites_fold = sorted(valid_df['site'].dropna().unique())
            
            print(f"Training: {len(train_df)} samples from {len(train_sites_fold)} sites")
            print(f"Validation: {len(valid_df)} samples from {len(valid_sites_fold)} sites")
            print(f"Validation sites: {valid_sites_fold}")

            # Calculate fold-specific class weights
            if USE_CLASS_WEIGHTS:
                fold_class_weights = calculate_class_weights(train_df, BAT_WEIGHT_MULTIPLIER)
            else:
                fold_class_weights = None

            # Create datasets
            train_ds = create_tf_dataset(train_df, IMG_HEIGHT, IMG_WIDTH,
                                        batch_size, is_training=True)
            valid_ds = create_tf_dataset(valid_df, IMG_HEIGHT, IMG_WIDTH,
                                        batch_size, is_training=False)

            # Build and initialise model
            tf.keras.backend.clear_session()
            model = create_resnet_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, channels),
                num_classes=len(CLASS_LABELS),
                learning_rate=FINE_TUNE_LR
            )
            model = load_pretrained_backbone_weights(model, pretrained_weights_path)

            # Head-only warmup phase
            run_head_start_phase(
                model, train_ds, valid_ds, fold_class_weights,
                fine_tune_lr=FINE_TUNE_LR, epochs=HEAD_START_EPOCHS
            )

            # Training callbacks
            early_stop = EarlyStopping(
                monitor='val_bat_pr_auc',
                patience=5,
                mode='max',
                restore_best_weights=True
            )
            
            ckpt = ModelCheckpoint(
                os.path.join(output_dir, f"fold{fold_num}_best.keras"),
                monitor='val_bat_pr_auc',
                mode='max',
                save_best_only=True
            )
            
            lr_schedule = LearningRateScheduler(
                make_lr_schedule(epochs, initial_lr=FINE_TUNE_LR)
            )

            # Train
            history = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=epochs,
                callbacks=[early_stop, ckpt, lr_schedule],
                class_weight=fold_class_weights
            )
            
            # Identify best epoch
            val_pr_auc_hist = history.history.get("val_bat_pr_auc", [])
            if val_pr_auc_hist:
                best_i = int(np.argmax(val_pr_auc_hist))
                best_val = float(val_pr_auc_hist[best_i])
                best_epoch = best_i + 1
            else:
                best_val = None
                best_epoch = epochs
                
            best_epochs.append(best_epoch)
            print(f"\nBest epoch: {best_epoch} (val_bat_pr_auc = {best_val:.4f})")

            # Evaluate on validation set
            print(f"\nValidation metrics:")
            eval_results = model.evaluate(valid_ds)
            for metric_name, value in zip(model.metrics_names, eval_results):
                print(f"  {metric_name}: {value:.4f}")

            # Collect predictions for out-of-fold aggregation
            y_true, y_prob = [], []
            for batch_images, batch_labels in valid_ds:
                prob = model.predict(batch_images, verbose=0)
                y_prob.append(prob)
                y_true.append(batch_labels.numpy())

            y_true = np.concatenate(y_true) if y_true else np.array([])
            y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, len(CLASS_LABELS)))

            if y_true.size == 0:
                print("[WARNING] No validation samples; skipping fold metrics")
                continue

            # Accumulate out-of-fold predictions
            oof_y_true.append(y_true)
            oof_y_prob.append(y_prob)

            # Per-fold evaluation
            y_pred = np.argmax(y_prob, axis=1)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS, normalize='true')
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                       xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Fold {fold_num} - Normalised Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fold{fold_num}_confusion_matrix.png"),
                       dpi=150)
            plt.close()

            # Classification report
            report = classification_report(
                y_true, y_pred,
                labels=CLASS_LABELS,
                target_names=SHORT_NAMES,
                output_dict=True,
                zero_division=0
            )
            
            print("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                labels=CLASS_LABELS,
                target_names=SHORT_NAMES,
                zero_division=0
            ))
            
            # Compute F2 scores
            f2_per_class = fbeta_score(y_true, y_pred, labels=CLASS_LABELS,
                                       beta=2.0, average=None, zero_division=0)
            f2_macro = fbeta_score(y_true, y_pred, labels=CLASS_LABELS,
                                   beta=2.0, average='macro', zero_division=0)
            f2_weighted = fbeta_score(y_true, y_pred, labels=CLASS_LABELS,
                                      beta=2.0, average='weighted', zero_division=0)

            # Compute per-class false positive rates
            cm_counts = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
            fp = cm_counts.sum(axis=0) - np.diag(cm_counts)
            fn = cm_counts.sum(axis=1) - np.diag(cm_counts)
            tp = np.diag(cm_counts)
            tn = cm_counts.sum() - (fp + fn + tp)
            fpr_per_class = fp / (fp + tn + 1e-8)

            # Store per-class metrics for fold-level standard deviation
            for idx, name in enumerate(SHORT_NAMES):
                cls_stats = report.get(name, {})
                fold_class_stats.append({
                    "fold": fold_num,
                    "class": name,
                    "support": int(cls_stats.get("support", 0)),
                    "precision": cls_stats.get("precision", 0.0),
                    "recall": cls_stats.get("recall", 0.0),
                    "f1": cls_stats.get("f1-score", 0.0),
                    "f2": f2_per_class[idx] if idx < len(f2_per_class) else 0.0,
                    "fpr": fpr_per_class[idx] if idx < len(fpr_per_class) else 0.0,
                })

            # Store macro/weighted metrics
            macro_stats = report.get("macro avg", {})
            weighted_stats = report.get("weighted avg", {})
            fold_avg_stats.append({
                "fold": fold_num,
                "avg_type": "macro",
                "precision": macro_stats.get("precision", 0.0),
                "recall": macro_stats.get("recall", 0.0),
                "f1": macro_stats.get("f1-score", 0.0),
                "f2": f2_macro,
            })
            fold_avg_stats.append({
                "fold": fold_num,
                "avg_type": "weighted",
                "precision": weighted_stats.get("precision", 0.0),
                "recall": weighted_stats.get("recall", 0.0),
                "f1": weighted_stats.get("f1-score", 0.0),
                "f2": f2_weighted,
            })

            # Record fold summary
            fold_metrics.append({
                "fold": fold_num,
                "train_sites": ", ".join(train_sites_fold),
                "val_sites": ", ".join(valid_sites_fold),
                "best_epoch": best_epoch,
                "val_bat_pr_AUC": best_val
            })

            tf.keras.backend.clear_session()

        # Save fold-level metrics
        fold_metrics_df = pd.DataFrame(fold_metrics)
        fold_metrics_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index=False)

        # ---------------------------------------------------------------------
        # OUT-OF-FOLD AGGREGATED EVALUATION
        # ---------------------------------------------------------------------
        if oof_y_true and oof_y_prob:
            print("\n" + "=" * 60)
            print("OUT-OF-FOLD AGGREGATED EVALUATION")
            print("=" * 60)
            
            y_true_oof = np.concatenate(oof_y_true)
            y_prob_oof = np.concatenate(oof_y_prob)
            y_pred_oof = np.argmax(y_prob_oof, axis=1)

            # OOF confusion matrix
            cm_oof = confusion_matrix(y_true_oof, y_pred_oof, labels=CLASS_LABELS,
                                     normalize='true')
            cm_oof_counts = confusion_matrix(y_true_oof, y_pred_oof, labels=CLASS_LABELS)
            
            fp_oof = cm_oof_counts.sum(axis=0) - np.diag(cm_oof_counts)
            fn_oof = cm_oof_counts.sum(axis=1) - np.diag(cm_oof_counts)
            tp_oof = np.diag(cm_oof_counts)
            tn_oof = cm_oof_counts.sum() - (fp_oof + fn_oof + tp_oof)
            fpr_oof = fp_oof / (fp_oof + tn_oof + 1e-8)
            support_oof = cm_oof_counts.sum(axis=1)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_oof, annot=True, fmt=".2f", cmap="Blues",
                       xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Out-of-Fold Confusion Matrix (argmax)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "oof_confusion_matrix.png"), dpi=150)
            plt.close()

            # OOF classification metrics
            report_oof = classification_report(
                y_true_oof, y_pred_oof,
                labels=CLASS_LABELS,
                target_names=SHORT_NAMES,
                output_dict=True,
                zero_division=0
            )
            
            f2_per_class = fbeta_score(y_true_oof, y_pred_oof, labels=CLASS_LABELS,
                                       beta=2.0, average=None, zero_division=0)
            f2_macro = fbeta_score(y_true_oof, y_pred_oof, labels=CLASS_LABELS,
                                   beta=2.0, average='macro', zero_division=0)
            f2_weighted = fbeta_score(y_true_oof, y_pred_oof, labels=CLASS_LABELS,
                                      beta=2.0, average='weighted', zero_division=0)

            # OOF PR curves
            present_labels_oof = np.unique(y_true_oof)
            _, curves_pr_oof, _, aps_oof = compute_multiclass_curves(
                y_true_oof, y_prob_oof, present_labels_oof
            )
            
            plt.figure(figsize=(7, 6))
            for c in CLASS_LABELS:
                if c in present_labels_oof and curves_pr_oof.get(c) is not None:
                    r, p = curves_pr_oof[c]
                    plt.plot(r, p, lw=1.5, label=f"{SHORT_NAMES[c]} (AP={aps_oof[c]:.2f})")
            if curves_pr_oof.get('micro') is not None:
                r_micro, p_micro = curves_pr_oof['micro']
                plt.plot(r_micro, p_micro, lw=2, linestyle='--',
                        label=f"micro (AP={aps_oof['micro']:.2f})")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("Out-of-Fold Precision-Recall Curves")
            plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "oof_pr_curves.png"), dpi=150)
            plt.close()

            # Compute fold-level standard deviations
            if fold_class_stats:
                fold_class_df = pd.DataFrame(fold_class_stats)
                class_std_df = fold_class_df.groupby("class").agg(
                    precision_std=("precision", "std"),
                    recall_std=("recall", "std"),
                    f1_std=("f1", "std"),
                    f2_std=("f2", "std"),
                    fpr_std=("fpr", "std"),
                ).reset_index()
                class_std_map = class_std_df.set_index("class").to_dict()
            else:
                class_std_map = {}

            # Build comprehensive results table
            report_rows = []
            for idx, name in enumerate(SHORT_NAMES):
                cls_stats = report_oof.get(name, {})
                stds = {
                    "precision_std": class_std_map.get("precision_std", {}).get(name, np.nan),
                    "recall_std": class_std_map.get("recall_std", {}).get(name, np.nan),
                    "f1_std": class_std_map.get("f1_std", {}).get(name, np.nan),
                    "f2_std": class_std_map.get("f2_std", {}).get(name, np.nan),
                    "fpr_std": class_std_map.get("fpr_std", {}).get(name, np.nan),
                }
                report_rows.append({
                    "class": name,
                    "support": int(cls_stats.get("support", 0)),
                    "precision": cls_stats.get("precision", 0.0),
                    "recall": cls_stats.get("recall", 0.0),
                    "f1": cls_stats.get("f1-score", 0.0),
                    "f2": f2_per_class[idx] if idx < len(f2_per_class) else 0.0,
                    "fpr": fpr_oof[idx] if idx < len(fpr_oof) else 0.0,
                    "precision_std": stds["precision_std"],
                    "recall_std": stds["recall_std"],
                    "f1_std": stds["f1_std"],
                    "f2_std": stds["f2_std"],
                    "fpr_std": stds["fpr_std"],
                })

            # Compute macro/weighted standard deviations
            if fold_avg_stats:
                fold_avg_df = pd.DataFrame(fold_avg_stats)
                avg_std_df = fold_avg_df.groupby("avg_type").agg(
                    precision_std=("precision", "std"),
                    recall_std=("recall", "std"),
                    f1_std=("f1", "std"),
                    f2_std=("f2", "std"),
                ).reset_index()
                avg_std_map = avg_std_df.set_index("avg_type").to_dict()
            else:
                avg_std_map = {}

            # Add macro and weighted averages
            macro_stats = report_oof.get("macro avg", {})
            weighted_stats = report_oof.get("weighted avg", {})
            total_support = int(macro_stats.get("support", len(y_true_oof)))
            fpr_macro = float(np.mean(fpr_oof))
            weights = support_oof / support_oof.sum() if support_oof.sum() > 0 else \
                      np.full_like(support_oof, np.nan, dtype=float)
            fpr_weighted = float(np.nansum(fpr_oof * weights))

            report_rows.append({
                "class": "macro",
                "support": total_support,
                "precision": macro_stats.get("precision", 0.0),
                "recall": macro_stats.get("recall", 0.0),
                "f1": macro_stats.get("f1-score", 0.0),
                "f2": f2_macro,
                "fpr": fpr_macro,
                "precision_std": avg_std_map.get("precision_std", {}).get("macro", np.nan),
                "recall_std": avg_std_map.get("recall_std", {}).get("macro", np.nan),
                "f1_std": avg_std_map.get("f1_std", {}).get("macro", np.nan),
                "f2_std": avg_std_map.get("f2_std", {}).get("macro", np.nan),
                "fpr_std": np.nan,
            })
            report_rows.append({
                "class": "weighted",
                "support": int(weighted_stats.get("support", len(y_true_oof))),
                "precision": weighted_stats.get("precision", 0.0),
                "recall": weighted_stats.get("recall", 0.0),
                "f1": weighted_stats.get("f1-score", 0.0),
                "f2": f2_weighted,
                "fpr": fpr_weighted,
                "precision_std": avg_std_map.get("precision_std", {}).get("weighted", np.nan),
                "recall_std": avg_std_map.get("recall_std", {}).get("weighted", np.nan),
                "f1_std": avg_std_map.get("f1_std", {}).get("weighted", np.nan),
                "f2_std": avg_std_map.get("f2_std", {}).get("weighted", np.nan),
                "fpr_std": np.nan,
            })

            # Save OOF classification report
            oof_report_df = pd.DataFrame(report_rows)
            oof_report_df.to_csv(os.path.join(output_dir, "oof_classification_report.csv"),
                                index=False)
            
            print("\nOut-of-Fold Classification Report:")
            print(oof_report_df.to_string(index=False))
            
            if "accuracy" in report_oof:
                print(f"\nOOF Accuracy: {report_oof['accuracy']:.4f}")

            # -----------------------------------------------------------------
            # THRESHOLD CALIBRATION
            # -----------------------------------------------------------------
            print("\n" + "-" * 40)
            print("BAT THRESHOLD CALIBRATION")
            print("-" * 40)
            
            p_bat_oof = y_prob_oof[:, BAT_ID]
            
            if THRESHOLD_STRATEGY == 'precision_constrained':
                bat_thresh_global = pick_optimal_threshold(
                    y_true_oof, p_bat_oof,
                    strategy='precision_constrained',
                    target_precision=TARGET_PRECISION
                )
            elif THRESHOLD_STRATEGY == 'recall_constrained':
                bat_thresh_global = pick_optimal_threshold(
                    y_true_oof, p_bat_oof,
                    strategy='recall_constrained',
                    target_recall=TARGET_RECALL
                )
            else:
                bat_thresh_global = pick_optimal_threshold(
                    y_true_oof, p_bat_oof,
                    strategy='fbeta',
                    beta=THRESHOLD_BETA
                )

            # Save threshold
            with open(os.path.join(output_dir, "bat_threshold_oof.txt"), "w") as f:
                f.write(str(bat_thresh_global))

            # Generate threshold analysis plots
            plot_threshold_analysis(
                y_true_oof, p_bat_oof, bat_thresh_global,
                os.path.join(output_dir, "oof_threshold_analysis.png"),
                bat_id=BAT_ID, beta=THRESHOLD_BETA
            )

            # Bat vs Other binary evaluation at calibrated threshold
            y_true_bat_oof = (y_true_oof == BAT_ID).astype(int)
            y_pred_bat_oof = (p_bat_oof >= bat_thresh_global).astype(int)

            cm_bat_oof = confusion_matrix(y_true_bat_oof, y_pred_bat_oof, normalize='true')
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_bat_oof, annot=True, fmt=".2f", cmap="Blues",
                       xticklabels=["other", "bat"], yticklabels=["other", "bat"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"OOF - Bat Detection (threshold={bat_thresh_global:.3f})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "oof_confusion_bat.png"), dpi=150)
            plt.close()
            
            # Binary classification report
            report_bat_oof = classification_report(
                y_true_bat_oof, y_pred_bat_oof,
                target_names=["other", "bat"],
                output_dict=True,
                zero_division=0
            )
            bat_fbeta_oof = fbeta_score(y_true_bat_oof, y_pred_bat_oof,
                                        beta=THRESHOLD_BETA, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true_bat_oof, y_pred_bat_oof,
                                              labels=[0, 1]).ravel()
            total = tn + fp + fn + tp
            prevalence = (tp + fn) / total if total else np.nan
            recall_bat = tp / (tp + fn) if (tp + fn) else 0.0

            pd.DataFrame([{
                "threshold": bat_thresh_global,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support_pos": tp + fn,
                "support_neg": tn + fp,
                "prevalence": prevalence,
                "bat_precision": report_bat_oof.get("bat", {}).get("precision", np.nan),
                "bat_recall": recall_bat,
                "bat_f1": report_bat_oof.get("bat", {}).get("f1-score", np.nan),
                "bat_f2": bat_fbeta_oof,
            }]).to_csv(os.path.join(output_dir, "oof_threshold_report.csv"), index=False)
        
        # Save optimal epoch count (median across folds)
        if best_epochs:
            optimal_epochs = int(np.median(best_epochs))
            with open(os.path.join(output_dir, "optimal_epochs.txt"), "w") as f:
                f.write(str(optimal_epochs))
            print(f"\nMedian optimal epochs across folds: {optimal_epochs}")

    # -------------------------------------------------------------------------
    # FINAL MODEL TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING HOLDOUT EVALUATION MODEL")
    print("=" * 60)
    
    # Load optimal epochs if available
    optimal_epochs_path = os.path.join(output_dir, "optimal_epochs.txt")
    if os.path.exists(optimal_epochs_path):
        with open(optimal_epochs_path, "r") as f:
            epochs_for_final = int(f.read().strip())
    else:
        epochs_for_final = epochs
    
    print(f"Using {epochs_for_final} epochs (from cross-validation)")
    
    holdout_model = train_final_model(
        df_train,
        os.path.join(output_dir, "holdout_model.keras"),
        IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH,
        channels=channels, batch_size=batch_size,
        epochs=epochs_for_final,
        pretrained_weights=pretrained_weights_path
    )

    # -------------------------------------------------------------------------
    # OPTIONAL: PRODUCTION MODEL ON ALL DATA
    # -------------------------------------------------------------------------
    if TRAIN_PRODUCTION_MODEL:
        print("\n" + "=" * 60)
        print("TRAINING PRODUCTION MODEL (ALL DATA)")
        print("=" * 60)
        
        production_model = train_final_model(
            df,
            os.path.join(output_dir, "production_model.keras"),
            IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH,
            channels=channels, batch_size=batch_size,
            epochs=epochs_for_final,
            pretrained_weights=pretrained_weights_path
        )

    # -------------------------------------------------------------------------
    # COMPLETION SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - fold_metrics.csv: Per-fold training summaries")
    print("  - oof_classification_report.csv: Aggregated cross-validation metrics")
    print("  - bat_threshold_oof.txt: Calibrated bat detection threshold")
    print("  - oof_threshold_analysis.png: Threshold selection diagnostics")
    print("  - holdout_model.keras: Model for holdout evaluation")
    if TRAIN_PRODUCTION_MODEL:
        print("  - production_model.keras: Final production model")


if __name__ == '__main__':
    main()