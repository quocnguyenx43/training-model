import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

from datasets import load_metric

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

    print(f'Evaluation, Loss: {running_loss:.4f}, ', end="")
    
    if task_running == 'task-1':
        show_evaluation_task_1(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_1(true_labels, predictions)
    elif task_running == 'task-2':
        show_evaluation_task_2(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_2(true_labels, predictions)

    return running_loss


def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, saving_path=None, task_running='task-1', device='cpu'):
    if task_running == 'task-1':
        dimesion = 1
    elif task_running == 'task-2':
        dimesion = 2
    
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
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
        val_running_loss = evaluate(model, criterion, dev_dataloader, cm=False, cr=False, last_epoch=False, task_running=task_running, device=device)

        # Saving
        if saving_path:
            if val_running_loss < best_val_loss:
                path = saving_path + "_" + str(epoch) + '.pth'
                torch.save(model.state_dict(), path)
                print('Saved the best model to path: ' +  path)
                best_val_loss = val_running_loss
                patience_counter = 0
            else:
                patience_counter += 1

        # Check if
        if patience_counter >= patience:
            print('Early stopping triggered')
            break
        
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
    class_names = ['clean', 'warning', 'seeding'] # 0, 1, 2
    cm = confusion_matrix(true_labels, predictions)#, labels=class_names)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(true_labels, predictions))#, target_names=class_names)


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


def generate_task_3(model, tokenizer, dataloader, target_len=512, device='cpu'):

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        # Wrap the dataloader with tqdm to monitor progress
        with tqdm(dataloader, desc="Evaluation") as tqdm_loader:
            for batch_idx, data in enumerate(tqdm_loader, 0):
                tqdm_loader.set_description(f"Validation, Batch {batch_idx + 1}/{len(dataloader)}")

                ids = data['input']['input_ids'].to(device, dtype=torch.long)
                mask = data['input']['attention_mask'].to(device, dtype=torch.long)
                y = data['label']['input_ids'].to(device, dtype=torch.long)

                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=target_len,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

                predictions.extend(preds)
                references.extend(target)
                
    return predictions, references


def train_task_3(model, optimizer, tokenizer, epochs, train_dataloader, dev_dataloader, target_len, saving_path=None, device='cpu'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap the train_dataloader with tqdm to monitor progress
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

                ids = batch['input']['input_ids'].to(device, dtype=torch.long)
                mask = batch['input']['attention_mask'].to(device, dtype=torch.long)

                y = batch['label']['input_ids'].to(device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=ids, attention_mask=mask,
                    decoder_input_ids=y_ids, labels=lm_labels
                )
                loss = outputs[0]
                running_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions, references = generate_task_3(model, tokenizer, dev_dataloader, target_len=target_len, device=device)
        metric = compute_score_task_3(predictions, references)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Rouge: {metric}')
        print()

        if saving_path:
            path = saving_path + "_" + str(epoch) + '.pth'
            torch.save(model.state_dict(), path)
            print('Saved the best model to path: ' +  path)
        
    print()


def compute_score_task_3(predictions, references):
    bertscore_metric = load_metric('bertscore')
    bertscore_result = bertscore_metric.compute(predictions=predictions, references=references, lang="vi")
    return bertscore_result