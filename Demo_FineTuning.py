import os
import torch
import logging
import numpy as np
from zipfile import ZipFile
from huggingface_hub import snapshot_download
from BudaOCR.Modules import EasterNetwork, OCRTrainer, WylieEncoder
from BudaOCR.Utils import create_dir, shuffle_data, build_data_paths, read_ctc_model_config


logging.getLogger().setLevel(logging.INFO)
print(torch.__version__)
torch.cuda.empty_cache()


# data_path = snapshot_download(repo_id="BDRC/Karmapa8", repo_type="dataset",  cache_dir="Datasets")
#
# with ZipFile(f"{data_path}/data.zip", 'r') as zip:
#     zip.extractall(f"{data_path}/Dataset")
#
# dataset_path = f"{data_path}/Dataset"

dataset_path = "train/"

image_paths, label_paths = build_data_paths(dataset_path)
image_paths, label_paths = shuffle_data(image_paths, label_paths)

print(f"Images: {len(image_paths)}, Labels: {len(label_paths)}")

model_path = snapshot_download(repo_id="BDRC/BigUCHAN_v1", repo_type="model",  cache_dir="Models")

model_config = f"{model_path}/config.json"
ctc_config = read_ctc_model_config(model_config)
label_encoder = WylieEncoder(ctc_config.charset)
num_classes = label_encoder.num_classes()

image_width = ctc_config.input_width
image_height = ctc_config.input_height

output_dir = "Output"
create_dir(output_dir)

network = EasterNetwork(num_classes=num_classes, image_width=ctc_config.input_width, image_height=ctc_config.input_height, mean_pooling=True)
workers = 4
batch_size = 16

checkpoint_path = f"{model_path}/BigUCHAN_E_v1.pth"
#network.load_model(checkpoint_path) # just load the weights
network.fine_tune(checkpoint_path) # load weights and freeze parts of the network

ocr_trainer = OCRTrainer(
    network=network,
    label_encoder=label_encoder,
    workers=workers,
    image_width=ctc_config.input_width,
    image_height=ctc_config.input_height,
    batch_size=batch_size,
    output_dir=output_dir,
    preload_labels=True
    )
ocr_trainer.init(image_paths, label_paths, train_split=0.8)

# adjust number of epochs and scheduler start based on the dataset size, smaller datasets require more epochs
ocr_trainer.train(epochs=64, check_cer=True, export_onnx=True)

# Evaluate test set
cer_scores = ocr_trainer.evaluate()
cer_values = list(cer_scores.values())

score_file = os.path.join(ocr_trainer.output_dir, "cer_scores.txt")

with open(score_file, "w", encoding="utf-8") as f:
    for sample, value in cer_scores.items():
        f.write(f"{sample} - {value}\n")

cer_summary_file = os.path.join(ocr_trainer.output_dir, "cer_summary.txt")

mean_cer = np.mean(cer_values)
max_cer = np.max(cer_values)
min_cer = np.min(cer_values)

with open(cer_summary_file, "w", encoding="utf-8") as f:
    f.write(f"Mean CER: {mean_cer}\n")
    f.write(f"Max CER: {max_cer}\n")
    f.write(f"Min CER: {min_cer}")

# export explicitly to onnx
network.export_onnx(out_dir=f"{ocr_trainer.output_dir}", model_name="OCRModel")