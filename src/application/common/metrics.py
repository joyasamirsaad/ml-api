from fastapi import UploadFile, File
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


async def metrics(file: UploadFile):
    if not file.filename.endswith(('.csv')):
        return {"error": "Invalid file type. Only CSV files are allowed."}
    
    # saving the uploaded file
    counter = 1
    c = 1
    if "tune" in file.filename:
        base_folder = Path(f"metrics/tuning{c}")
        folder_location = base_folder
        if folder_location.exists():
            c += 1
            folder_location = Path(f"metrics/tuning{c}")
    else:
        base_folder = Path(f"metrics/plots{counter}")
        folder_location = base_folder
        while folder_location.exists():
            counter += 1
            folder_location = Path(f"metrics/plots{counter}")

    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # reading the csv file
    df = pd.read_csv(file_location) 

    # plotting the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')  
    plt.title('Training and Validation Box Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/box_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='blue')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Class Loss')    
    plt.title('Training and Validation Class Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/class_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/dfl_loss'], label='Train dfl Loss', color='blue')
    plt.plot(df['epoch'], df['val/dfl_loss'], label='Val dfl Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('dfl Loss')
    plt.title('Training and Validation dfl Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/dfl_loss.png')
    plt.show()
    plt.close()

    x = np.arange(len(df['epoch']))  
    bar_width = 0.2
    metrics = [
        ('metrics/mAP50(B)', 'mAP50', 'green'),
        ('metrics/mAP50-95(B)', 'mAP50-95', 'orange'),
        ('metrics/precision(B)', 'Precision', 'purple'),
        ('metrics/recall(B)', 'Recall', 'brown')
    ]
    for i, (col, label, color) in enumerate(metrics):
        plt.bar(x+i*bar_width, df[col], width=bar_width, label=label, color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('mAP, Precision, and Recall over Epochs')
    plt.xticks(x+bar_width*(len(metrics)-1)/2, df['epoch'])
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/mAP_precision_recall_bar.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['epoch'], df['lr/pg0'], label='Learning Rate pg0', color='purple', marker='o')
    plt.scatter(df['epoch'], df['lr/pg1'], label='Learning Rate pg1', color='brown', marker='x')
    plt.scatter(df['epoch'], df['lr/pg2'], label='Learning Rate pg2', color='pink', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/learning_rate.png') 
    plt.show() 
    plt.close()