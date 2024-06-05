import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import utils


def evaluate(model, criterion, dataloader, task_running='task-1', cm=False, cr=False, last_epoch=False, device='cpu'):
    if task_running == 'task-1':
        dimesion = 1
    elif task_running == 'task-2':
        dimesion = 2

    running_loss = 0.0
    predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        # Wrap the dataloader with tqdm to monitor progress
        with tqdm(dataloader, desc="Evaluation") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Evaluation, Batch {batch_idx + 1}/{len(dataloader)}")

                inputs = batch['input'].to(device)
                label = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = torch.max(outputs, dim=dimesion)
                _, true = torch.max(label, dim=dimesion)

                # Append predictions and true labels for F1 score calculation
                predictions.extend(pred.tolist())
                true_labels.extend(true.tolist())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    if task_running == 'task-1':
        show_evaluation_task_1(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_1(true_labels, predictions)
    elif task_running == 'task-2':
        show_evaluation_task_2(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_2(true_labels, predictions)


def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, saving_path=None, task_running='task-1', device='cpu'):
    if task_running == 'task-1':
        dimesion = 1
    elif task_running == 'task-2':
        dimesion = 2

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        true_labels = []
        predictions = []

        # Wrap the train_dataloader with tqdm to monitor progress
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

                inputs = batch['input'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, pred = torch.max(outputs, dim=dimesion)
                _, true = torch.max(label, dim=dimesion)
                
                # Append predictions and true labels for metrics calculation
                true_labels.extend(true.tolist())
                predictions.extend(pred.tolist())

        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, ', end='')

        if task_running == 'task-1':
            show_evaluation_task_1(true_labels, predictions)
        elif task_running == 'task-2':
            show_evaluation_task_2(true_labels, predictions)

        # Evaluation on Dev set
        evaluate(model, criterion, dev_dataloader, cm=False, cr=False, last_epoch=False, task_running=task_running)

        if saving_path:
            path = saving_path + "_" + str(epoch) + '.pth'
            torch.save(model.state_dict(), path)
            print('Saved Model in ' +  path)

        print()


def show_evaluation_task_1(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    prec = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    
    print(
        f'Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
    )


def show_cm_cr_task_1(true_labels, predictions):
    class_names = ['clean', 'warning', 'seeding']  # 0, 1, 2
    cm = confusion_matrix(true_labels, predictions)#, labels=class_names)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(true_labels, predictions))#, target_names=class_names))


def show_evaluation_task_2(true_labels, predictions):
    zero_one_loss = np.any(true_labels != predictions, axis=1).mean()
    hamming_loss = utils.my_hamming_loss(true_labels, predictions)
    emr = np.all(predictions == true_labels, axis=1).mean()

    acc = utils.my_accuracy(true_labels, predictions)
    prec = utils.my_precision(true_labels, predictions)
    f1 = utils.my_f1_score(true_labels, predictions)
    recall = utils.my_recall(true_labels, predictions)

    print(
        f'0/1 Loss: {zero_one_loss:.4f}, Hamming Loss: {hamming_loss:.4f}, EMR: {emr:.4f}, ' \
        + f'Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}'
    )


def show_cm_cr_task_2(true_labels, predictions):
    aspect_names = ['title', 'desc', 'company', 'other']
    for i, aspect in enumerate(aspect_names):
        cm_p = confusion_matrix(true_labels[:, i], predictions[:, i])#, labels=['neu', 'pos', 'neg', 'nm'])
        print(f"Confusion Matrix of {aspect} aspect")
        print(cm_p)
        print(f"Classification Report for {aspect} aspect")
        print(classification_report(true_labels[:, i], predictions[:, i]))#, target_names=['neu', 'pos', 'neg', 'nm'])