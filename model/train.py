import model.cnn as cnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import OutputModel, Task
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torchvision import transforms
from data.data_downloader import load_dataset


def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def eval_metrics(net, data_loader, num_classes=38, topk=5):
    net.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(get_device()), data[1].to(get_device())
            outputs = net(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds)
            y_prob.append(probs)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    out = {"acc": float(accuracy_score(y_true, y_pred))}
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out.update(
        {
            "p_micro": float(p_micro),
            "r_micro": float(r_micro),
            "f1_micro": float(f1_micro),
            "p_macro": float(p_macro),
            "r_macro": float(r_macro),
            "f1_macro": float(f1_macro),
        }
    )
    try:
        out["cm"] = confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist()
    except Exception:
        pass
    if topk and topk > 1:
        try:
            out[f"top{topk}"] = float(
                top_k_accuracy_score(
                    y_true,
                    np.concatenate(y_prob),
                    k=topk,
                    labels=list(range(num_classes)),
                )
            )
        except Exception:
            pass
    return out


# Run a network using the given DataLoader
# Returns the total accuracy
def run(net, data_loader):
    correct, total = 0, 0
    net.eval()  # switch the model to evaluation mode
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(get_device()), data[1].to(get_device())
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Train a network using the given training and validation DataLoaders
# Returns the final validation accuracy
def train(net, train_loader, train_loader_clean, val_loader, logger, lr, opt=optim.AdamW, max_epoch=50, num_classes=38, class_names=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = opt(net.parameters(), lr=lr)
    
    topk = 5
    best_accuracy = 0
    best_epoch = 0

    for epoch in range(1, max_epoch + 1):
        net.train()  # switch model to training mode
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(get_device()), data[1].to(get_device())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Compute metrics
        train_m = eval_metrics(net, train_loader_clean, num_classes=num_classes, topk=topk)
        val_m = eval_metrics(net, val_loader, num_classes=num_classes, topk=topk)
        accuracy = val_m["acc"]
        
        
        # We only log macro as micro is redundant (it's the same as accuracy)
        logger.report_scalar(title="Accuracy", series="train", value=train_m["acc"], iteration=epoch)
        logger.report_scalar(title="Precision", series="train", value=train_m["p_macro"], iteration=epoch)
        logger.report_scalar(title="Recall", series="train", value=train_m["r_macro"], iteration=epoch)
        logger.report_scalar(title="F1", series="train", value=train_m["f1_macro"], iteration=epoch)
        if "top5" in train_m:
            logger.report_scalar(title="TopK", series="top5_train", value=train_m["top5"], iteration=epoch)

        logger.report_scalar(title="Accuracy", series="val", value=val_m["acc"], iteration=epoch)
        logger.report_scalar(title="Precision", series="val", value=val_m["p_macro"], iteration=epoch)
        logger.report_scalar(title="Recall", series="val", value=val_m["r_macro"], iteration=epoch)
        logger.report_scalar(title="F1", series="val", value=val_m["f1_macro"], iteration=epoch)
        if "top5" in val_m:
            logger.report_scalar(title="TopK", series="top5_val", value=val_m["top5"], iteration=epoch)

        avg_loss = running_loss / len(train_loader)

        print(f"[{epoch}] val_acc: {accuracy:.4f}, train_acc: {train_m['acc']:.4f}, train_loss: {avg_loss:.4f}")

        # Log loss
        logger.report_scalar( title="Loss", series="train", value=avg_loss, iteration=epoch)

        if accuracy > best_accuracy:
            if "cm" in val_m:
                logger.report_confusion_matrix(
                    title="Confusion Matrix",
                    series="val",
                    matrix=val_m["cm"],
                    xlabels=[str(i) for i in range(num_classes)], 
                    ylabels=[str(i) for i in range(num_classes)],  
                    iteration=epoch
                )
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(net.state_dict(), "best_model.pt")
                

            print(f"Epoch {epoch}, new best model found (validation accuracy: {best_accuracy:.4f})")

        # Finish if validation accuracy is not improving
        if epoch - best_epoch >= 10:
            print(f"Early stopping triggered: no improvement for 10 epochs")
            break

    # Log final test accuracy
    logger.report_single_value(name="Final Validation Accuracy", value=best_accuracy)

    return best_accuracy


def test_general():
    task = Task.init(
        task_name="CNN_11-General", project_name="Small Group Project - Team 2"
    )
    
    #task.execute_remotely(queue_name="default")
    
    lr, batch_size, dropout, max_epoch, dataset, version = 0.0001, 32, 0.2, 100, 'plant_village_processed', '1.0.8'
    data_loader, data_loader_clean, val_loader, test_loader, mean, std = load_dataset(dataset, batch_size, 'processed_dataset_general')
    
    # Get class names from the dataset
    train_dataset = data_loader.dataset
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    elif hasattr(train_dataset, 'class_to_idx'):
        # Create list from class_to_idx dict, sorted by index
        class_names = [name for name, idx in sorted(train_dataset.class_to_idx.items(), key=lambda x: x[1])]
    else:
        # Fallback: infer from folder structure or use default
        class_names = [str(i) for i in range(38)]  # or detect num_classes dynamically
    
    num_classes = len(class_names)
    print(num_classes)
    param = {
        # Training hyperparameters
        "epochs": max_epoch,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "early_stopping_patience": 10,
        
        # Model architecture hyperparameters
        "model": "CNN_11",
        "input_resolution": (256, 256),
        "num_classes": num_classes,
        "dropout_rate": 0.2,  # Make sure this matches what you pass to CNN_10!
        "init_outputs": 16,
        "kernel_sizes": [3,3,3,3,3,3,3],
        "linear_layer_sizes": [128, 64],
        "pretrained": False,
        
        # Dataset hyperparameters
        "dataset": dataset,
        "dataset_version": version,
        "data_augmentation": True,
        "aug_random_crop_scale": (0.5, 1.0),
        "aug_color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        "aug_gaussian_blur": {"kernel_size": 5, "sigma": (0.1, 2.0)},
        "normalization": f"mean={mean}, std={std}",
        "data_split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    
    task.connect(param)
    logger = task.get_logger()

    # Create cnn
    net = cnn.CNN_11(
        categories=num_classes,
        dropout=dropout
    ).to(get_device())

    # Train it and test it (pass num_classes and class_names)
    val_accuracy = train(net, data_loader, data_loader_clean, val_loader, logger, lr=lr, num_classes=num_classes, class_names=class_names, max_epoch=max_epoch)

    # Load and evaluate best model
    best_model = cnn.CNN_11(categories=num_classes, dropout=dropout).to(get_device())
    best_model.load_state_dict(torch.load("best_model.pt"))
    test_accuracy = run(best_model, test_loader)
    print(
        f"Validation accuracy: {val_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}"
    )

    # Log final test accuracy
    logger.report_single_value(name="Final Test Accuracy", value=test_accuracy)


if __name__ == "__main__":
    test_general()
