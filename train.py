import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from dataset import QuickDrawDataset, load_prepared_data, train_transform, val_transform
import utils
import wandb
import argparse

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, model_ema=None, model_ema_steps=32, lr_warmup_epochs=2, scaler=None, print_freq=10):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if model_ema and i % model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print(f"Epoch [{epoch}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc, "Epoch": epoch}, step=epoch)

def evaluate(model, criterion, data_loader, device, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    wandb.log({"Validation Loss": epoch_loss, "Validation Accuracy": epoch_acc, "Epoch": epoch}, step=epoch)

    return epoch_acc

def main(data_dir, config=None):
    with wandb.init(config=config):
        config = wandb.config

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        (train_images, train_labels), (val_images, val_labels), _, classes = load_prepared_data(data_dir)

        train_dataset = QuickDrawDataset(train_images, train_labels, transform=train_transform)
        val_dataset = QuickDrawDataset(val_images, val_labels, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = timm.create_model(
            config.model_name,
            pretrained=True,
            num_classes=len(classes),
            in_chans=1,
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        parameters = utils.set_weight_decay(
            model,
            config.weight_decay,
            norm_weight_decay=config.norm_weight_decay,
        )

        optimizer = optim.SGD(model.parameters(),
                            lr=config.learning_rate,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.num_epochs - config.lr_warmup_epochs
        )

        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.lr_warmup_decay, total_iters=config.lr_warmup_epochs
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.lr_warmup_epochs]
        )

        if config.model_ema_steps > 0:
            adjust = config.batch_size * config.model_ema_steps / config.batch_size
            alpha = 1.0 - config.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
        else:
            model_ema = None

        scaler = torch.cuda.amp.GradScaler()

        best_val_accuracy = 0.0

        for epoch in range(config.num_epochs):
            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, model_ema=model_ema, model_ema_steps=config.model_ema_steps, lr_warmup_epochs=config.lr_warmup_epochs, scaler=scaler)
            lr_scheduler.step()

            if model_ema:
                val_epoch_acc = evaluate(model_ema, criterion, val_loader, device, epoch)
            else:
                val_epoch_acc = evaluate(model, criterion, val_loader, device, epoch)

            # Save the model with the epoch number, model name, and accuracy
            torch.save(model.state_dict(), f"{config.model_name}_epoch_{epoch+1}_acc_{val_epoch_acc:.4f}.pth")
            print(f"Model saved as '{config.model_name}_epoch_{epoch+1}_acc_{val_epoch_acc:.4f}.pth'")

            # Save the best model
            if val_epoch_acc > best_val_accuracy:
                best_val_accuracy = val_epoch_acc
                torch.save(model.state_dict(), f"best_model_{config.model_name}_acc_{best_val_accuracy:.4f}.pth")
                print(f"Best model saved as 'best_model_{config.model_name}_acc_{best_val_accuracy:.4f}.pth'")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy
            }
            if model_ema:
                checkpoint['model_ema_state_dict'] = model_ema.state_dict()
            torch.save(checkpoint, f"checkpoint_{config.model_name}_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved as 'checkpoint_{config.model_name}_epoch_{epoch+1}.pth'")

        print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Draw Challenge Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "Validation Accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "model_name": {"values": ["efficientnet_b3", "resnet50d", "resnext50d_32x4d", "efficientnet_b2", "resnext101_32x4d"]},
            "batch_size": {"values": [64, 128, 256, 512]},
            "num_epochs": {"min": 10, "max": 100},
            "learning_rate": {"min": 0.001, "max": 0.5},
            "label_smoothing": {"value": 0.1},
            "momentum": {"min": 0.8, "max": 0.95},
            "weight_decay": {"values": [1e-4, 1e-5]},
            "norm_weight_decay": {"values": [0, 1e-5]},
            "model_ema_steps": {"values": [0, 32, 64]},
            "model_ema_decay": {"min": 0.97, "max": 0.999999},
            "lr_warmup_epochs": {"values": [2, 3, 4, 5]},
            "lr_warmup_decay": {"min": 0.001, "max": 0.3},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="quick_draw_challenge_2")
    wandb.agent(sweep_id, lambda: main(args.data_dir))
