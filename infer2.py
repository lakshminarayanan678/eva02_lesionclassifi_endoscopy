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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", ".*A new version of Albumentations is*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

def save_predictions_to_excel(image_paths, pred_probs, pred_classes, true_labels, class_names, save_path):
    class_columns = [f"prob_{c}" for c in class_names]
    predicted_class_names = [class_names[i] for i in pred_classes]
    true_class_names = [class_names[i] if i != -1 else "unknown" for i in true_labels]

    df = pd.DataFrame(pred_probs, columns=class_columns)
    df["image_path"] = image_paths
    df["predicted_class"] = predicted_class_names
    df["true_class"] = true_class_names
    df.to_excel(save_path, index=False)

def prepare_model(ckpt_path, config, class_to_idx):
    if not ckpt_path:
        raise ValueError("Checkpoint path not provided")
    logging.info(f"Loading model from checkpoint: {ckpt_path}")

    model = TimmModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        class_to_idx=class_to_idx
    )
    model.eval()
    return model

def load_data(csv_path, dataset_type, config):
    csv_file = os.path.join(csv_path, f"{dataset_type}.csv")
    df = pd.read_csv(csv_file)

    # Extract class name from image path
    df["true_class"] = df["frame_path"].apply(lambda x: x.split("/")[-2].lower())

    class_name_to_index = {name.lower(): idx for idx, name in enumerate(config.class_names)}
    image_paths = df["frame_path"].tolist()
    labels = [class_name_to_index.get(name, -1) for name in df["true_class"]]

    return image_paths, labels, df

def main(args):
    config_args = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
    for key, value in config_args.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    config = argparse.Namespace(**vars(args))
    config.ft_mode = None
    dataset_type = config.dataset_type

    # Load class mapping
    class_mapping = load_class_mapping(config.class_mapping_filename)
    class_names = list(class_mapping.keys())

    # Load transforms
    _, val_transforms = load_transforms(img_size=config.img_size, transform_path=config.transform_path)

    # Load data
    image_paths, true_labels, df = load_data(config.dataset_csv_path, dataset_type, config)

    # Create dataset and dataloader
    dataset = ImageDataset(list(zip(image_paths, true_labels)), transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=config.val_bs, shuffle=False, num_workers=4)

    # Load model
    checkpoint_path = os.path.join(config.pretrained_checkpoint_dir, config.checkpoint_filename)
    model = prepare_model(ckpt_path=checkpoint_path, config=config, class_to_idx=class_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logging.info(f"Predicting on {len(dataset)} images")

    # Inference loop
    preds = []
    confidences = []
    pred_classes = []
    img_paths = []
    true_labels_all = []

    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc="Running Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            preds.append(probs.cpu().numpy())
            pred_classes.extend(torch.argmax(probs, dim=1).cpu().numpy())
            confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())
            true_labels_all.extend(labels.cpu().numpy())
            img_paths.extend(paths)

    preds = np.vstack(preds)

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Save prediction distribution
    class_counts = Counter(pred_classes)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=[class_counts.get(i, 0) for i in range(len(class_names))])
    plt.title("Prediction Count per Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, f"predicted_class_distribution_{dataset_type}.png"))
    plt.close()

    # Save confidence histogram
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Top-1 Confidence Score")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, f"prediction_confidence_histogram_{dataset_type}.png"))
    plt.close()

    # Save least confident predictions
    df_conf = pd.DataFrame({
        "image_path": img_paths,
        "predicted_class": [class_names[i] for i in pred_classes],
        "confidence": confidences
    })
    df_conf = df_conf.sort_values(by="confidence").reset_index(drop=True)
    df_conf.to_excel(os.path.join(config.save_dir, f"least_confident_predictions_{dataset_type}.xlsx"), index=False)

    # Save all predictions
    output_path = os.path.join(config.save_dir, f"combined_predictions_{dataset_type}.xlsx")
    save_predictions_to_excel(
        image_paths=img_paths,
        pred_probs=preds,
        pred_classes=pred_classes,
        true_labels=true_labels_all,
        class_names=class_names,
        save_path=output_path
    )

    # Filter out unknown labels and compute confusion matrix
    valid = [i for i, label in enumerate(true_labels_all) if label != -1]
    y_true = [true_labels_all[i] for i in valid]
    y_pred = [pred_classes[i] for i in valid]

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, f"confusion_matrix_{dataset_type}.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--save_dir", type=str, default="submission")
    parser.add_argument("--val_bs", type=int, default=32)

    parser.add_argument("--checkpoint_filename", type=str)
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_csv_path", type=str)
    parser.add_argument("--dataset_type", default="test", type=str)
    parser.add_argument("--class_mapping_filename", type=str)
    parser.add_argument("--transform_path", type=str)

    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--lambda_factor", type=float)

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main(args)