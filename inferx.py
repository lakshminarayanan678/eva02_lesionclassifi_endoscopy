import argparse
import logging
import os

from tqdm import tqdm

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.data.dataset import ImageDataset
from src.models.timm.timm_model import TimmModel
from src.utils.class_mapping import load_class_mapping
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", ".*A new version of Albumentations is*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")


def save_predictions_to_excel(image_paths, y_pred, output_path, dataset_type):
    logging.info(f"Saving predictions.xlsx to {output_path}")
    class_columns = ['erosion', 'normal', 'ulcers']

    y_pred_classes = np.argmax(y_pred, axis=1)

    if dataset_type == 'test':
        predicted_class_names = [class_columns[i] for i in y_pred_classes]
        df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    else:
        # Extract dataset name from each image path
        dataset_names = [image_path.split('\\')[-2] for image_path in image_paths]
        df_class = pd.DataFrame({'image_path': image_paths, 'Dataset': dataset_names})

    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)

    df_merged = pd.merge(df_prob, df_class, on='image_path')

    # Rearrange columns to have 'Dataset' as the second column if it exists
    desired_columns = ['image_path']
    if 'Dataset' in df_merged.columns:
        desired_columns.append('Dataset')
    desired_columns.extend([col for col in df_merged.columns if col not in desired_columns])
    df_merged = df_merged[desired_columns]

    df_merged.to_excel(output_path, index=False)


def prepare_model(ckpt_path, config, class_to_idx):
    """Load the model from the checkpoint."""
    if not ckpt_path:
        raise ValueError("Checkpoint path not provided")
    logging.info(f"Loading model from checkpoint: {ckpt_path}")

    model = TimmModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        class_to_idx=class_to_idx
    )
    if not model:
        raise ValueError("Failed to load model from checkpoint")
    model.eval()
    return model

    # model = TimmModel(config=config, class_to_idx=class_to_idx)
    # state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]
        # # Remove classifier layer weights (they mismatch)
    # state_dict = {k: v for k, v in state_dict.items() if "classifier.linear" not in k}
    # # Load only matching layers
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    # model.eval()




def load_data(dataset_csv_path, dataset_path, dataset_type='test'):
    csv_path = os.path.join(dataset_csv_path, 'test.csv')
    if dataset_type == 'val' or dataset_type == 'train':
        csv_path = os.path.join(dataset_csv_path, 'train_val.csv')

    df = pd.read_csv(csv_path)

    if dataset_type == 'val':
        df = df[df['fold'] == 0]
    elif dataset_type == 'train':
        df = df[df['fold'] > 0]

    df['frame_path'] = df['frame_path'].str.replace("\\", "/", regex=False)
    df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(dataset_path, x))
    return df


def get_image_batch(dataset, start_idx, batch_size):
    max_idx = len(dataset)
    end_idx = min(start_idx + batch_size, max_idx)
    batch_images = []

    for i in range(start_idx, end_idx):
        # Assuming dataset returns only image when class_to_index is not provided
        image = dataset[i]
        batch_images.append(image)

    # Stack the list of images into tensors
    batch_images = torch.stack(batch_images)

    return batch_images


def main(args):
    config_args = {} 

    # Load config file parameters if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)

    # Ensure argparse arguments override config file arguments
    args_dict = vars(args)
    for key, value in config_args.items():
        # If the argparse argument was not provided (None), use the value from the config file
        if args_dict.get(key) is None:
            setattr(args, key, value)

    config = argparse.Namespace(**vars(args))

    config.ft_mode = None
    dataset_type = getattr(config, "dataset_type", "test")

    class_mapping = load_class_mapping("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/capsulevision/class_mapping1.json")

    _, val_transforms = load_transforms(img_size=config.img_size, transform_path="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/codes/capsule_vision_challenge_2024/configs/transforms/base_transforms.py")

    df = load_data(config.dataset_csv_path, config.dataset_path, dataset_type)

    # Initialize Dataset
    dataset = ImageDataset(df, transform=val_transforms)

    # Load Model
    try:
        checkpoint_filename = getattr(config, "checkpoint_filename", None)
        checkpoint_dir = getattr(config, "pretrained_checkpoint_dir", None)
        checkpoint_path = None
        if checkpoint_filename and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    except AttributeError:
        raise ValueError("Checkpoint filename and directory not provided.")

    model = prepare_model(ckpt_path=checkpoint_path, config=config, class_to_idx=class_mapping)
    print("class_to_idx:", class_mapping)
    
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = config.val_bs
    num_batches = (len(dataset) // batch_size) + 1

    remove_prefix = os.path.join(config.dataset_path, "capsulevision\\\\")
    df['frame_path'] = df['frame_path'].str.replace(f'^{remove_prefix}', '', regex=True)

    if dataset_type == 'test':
        img_paths = df['proposed_name'].values
    else:
        img_paths = df['frame_path'].values

    logging.info(f"Predicting on {len(dataset)} images in {num_batches} batches")

    preds = []
    for batch_num in tqdm(range(1, num_batches + 1), desc="Processing Batches", unit="batch"):
        current_start_idx = (batch_num - 1) * batch_size

        images = get_image_batch(dataset, start_idx=current_start_idx, batch_size=batch_size)

        if torch.cuda.is_available():
            images = images.cuda()

        # Forward pass
        with torch.no_grad():
            logits = model.forward(images)

        batch_preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds.append(batch_preds)

    preds = np.vstack(preds)
    preds = preds.reshape(-1, 3)
    # Get predicted class indices and confidence scores
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    # Class distribution bar plot
    class_counts = Counter(pred_classes)
    class_names = list(class_mapping.keys())
    print("Class counts:", class_counts)
    print("Class names:", class_names)
    print("Predicted classes:", pred_classes)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=[class_counts.get(i, 0) for i in range(len(class_names))])
    plt.title("Prediction Count per Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/predicted_class_distribution_{dataset_type}.png")
    plt.close()

    # Confidence histogram
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Top-1 Confidence Score")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/prediction_confidence_histogram_{dataset_type}.png")
    plt.close()

    # Save least confident predictions to Excel
    df_confidence = pd.DataFrame({
        'image_path': img_paths,
        'predicted_class': [class_names[i] for i in pred_classes],
        'confidence': confidences
    })
    df_confidence = df_confidence.sort_values(by='confidence').reset_index(drop=True)
    df_confidence.to_excel(f"{config.save_dir}/least_confident_predictions_{dataset_type}.xlsx", index=False)

    os.makedirs('submission', exist_ok=True)

    os.makedirs(config.save_dir, exist_ok=True)
    output_path = f'{config.save_dir}/comb_mln2_{dataset_type}_dataset.xlsx'

    save_predictions_to_excel(image_paths=img_paths, y_pred=preds, output_path=output_path, dataset_type=dataset_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--save_dir", type=str, default="submission", help="Directory to save the prediction.xlsx file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--val_bs", default=32, type=int, help="Batch size for prediction")

    # === Paths ===
    parser.add_argument("--checkpoint_filename", type=str)
    parser.add_argument("--pretrained_checkpoint_dir", type=str, help="Directory to load pretrained checkpoints from")

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_csv_path", type=str)
    parser.add_argument("--dataset_type", default="test", type=str, help="Type of dataset to load ('train', 'val', 'test')")
    parser.add_argument("--class_mapping_filename", default="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/capsulevision/class_mapping1.json", type=str)
    parser.add_argument("--transform_path", type=str)

    # === Model ===
    parser.add_argument("--img_size", default=224, type=int)

    # === Optimizer ===
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", default=2e-4, type=float)

    # === Scheduler ===
    parser.add_argument("--lambda_factor", type=float, help="Lambda for the scheduler")

    arguments = parser.parse_args()

    # Initialize logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    if arguments.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arguments)
