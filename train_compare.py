import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from config import (
    DATA_ROOT, BATCH_SIZE, EPOCH_NUM, DEVICE,
    LR_LIST, SAVE_MODEL_DIR, TB_LOG_ROOT
)

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    return fig

def run_single_lr(lr):
    log_dir = os.path.join(TB_LOG_ROOT, f"lr_{lr:.0e}")
    writer = SummaryWriter(log_dir=log_dir)

    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)
    model_save_path = os.path.join(SAVE_MODEL_DIR, f"mnist_lr_{lr:.0e}.pth")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MNISTCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_step = 0
    for epoch in range(EPOCH_NUM):

        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), total_step)
            total_step += 1

        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                test_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

        acc = correct / total
        avg_test_loss = test_loss / len(test_loader)
        writer.add_scalar("Test/Loss", avg_test_loss, epoch)
        writer.add_scalar("Test/Accuracy", acc, epoch)

        print(f"LR={lr:.0e} | Epoch:{epoch} | Test Acc:{acc:.4f}")

    torch.save({
        'epoch': EPOCH_NUM,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': lr,
    }, model_save_path)
    print(f"模型已保存至 {model_save_path}")

    class_names = [str(i) for i in range(10)]
    cm_fig = plot_confusion_matrix(all_labels, all_preds, class_names)
    writer.add_figure("Confusion Matrix", cm_fig)

    sample_imgs, _ = next(iter(train_loader))
    sample_imgs = sample_imgs.to(DEVICE)
    writer.add_graph(model, sample_imgs)
    writer.add_images("Train/Samples", sample_imgs[:16].cpu())

    writer.close()

def load_demo():
    demo_lr = LR_LIST[1]
    load_path = os.path.join(SAVE_MODEL_DIR, f"mnist_lr_{demo_lr:.0e}.pth")
    checkpoint = torch.load(load_path, map_location=DEVICE)
    load_model = MNISTCNN().to(DEVICE)
    load_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载 lr={checkpoint['lr']:.0e} 的模型，训练轮次：{checkpoint['epoch']}")

if __name__ == "__main__":
    for learning_rate in LR_LIST:
        run_single_lr(learning_rate)
    # load_demo()
    print(f"\n所有学习率实验完成！启动tensorboard命令：")
    print(f"tensorboard --logdir={TB_LOG_ROOT}") 
