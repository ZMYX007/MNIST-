import torch
DATA_ROOT = "./data"
BATCH_SIZE = 64
EPOCH_NUM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR_LIST = [1e-2, 1e-3, 1e-4]
SAVE_MODEL_DIR = "./saved_models"
TB_LOG_ROOT = "./logs/compare_lr"
