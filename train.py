from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import torch


def evaluate(model, criterion, dataloader, device='cpu', cm=False, cr=False, last_epoch=False):
    model.eval()

    running_loss = 0.0
    correct = 0.0
    total = 0
    predictions = []
    true_labels = []

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

                _, pred = torch.max(outputs, dim=1)
                _, true = torch.max(label, dim=1)

                correct += torch.sum(pred == true).item()
                total += label.size(0)

                # Append predictions and true labels for F1 score calculation
                predictions.extend(pred.tolist())
                true_labels.extend(true.tolist())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    epoch_precision = precision_score(true_labels, predictions, average='macro')
    epoch_recall = recall_score(true_labels, predictions, average='macro')
    epoch_f1 = f1_score(true_labels, predictions, average='macro')

    if last_epoch:
        print()

    print(
        f"Evaluation, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}\n"
    )

    # Confusion Matrix
    if cm:
        cm = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:")
        print(cm)

    # Classification Report
    if cr:
        class_names = ['clean', 'warning', 'seeding']  # 0, 1, 2
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))



def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, device='cpu', saving_path=False):
    model.train()

    total_step = len(train_dataloader)
    train_losses = []
    train_accs = []
    train_precisions = []
    train_recalls = []
    train_f1s = []

    for epoch in range(epochs):
        model.train()
        total = 0
        running_loss = 0.0
        correct = 0.0
        true_labels = []
        predictions = []

        # Wrap the train_dataloader with tqdm to monitor progress
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

                batch_correct = 0


                inputs = batch['input'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, pred = torch.max(outputs, dim=1)
                _, true = torch.max(label, dim=1)

                batch_correct += torch.sum(pred == true).item()
                correct += batch_correct
                total += label.size(0)

                # Append predictions and true labels for metrics calculation
                predictions.extend(pred.tolist())
                true_labels.extend(true.tolist())

        train_losses.append(running_loss / total_step)
        train_accs.append(100 * correct / total)
        train_f1s.append(f1_score(true_labels, predictions, average='macro'))
        train_precisions.append(precision_score(true_labels, predictions, average='macro'))
        train_recalls.append(recall_score(true_labels, predictions, average='macro'))
        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f}, Precision: {train_precisions[-1]:.4f}, Recall: {train_recalls[-1]:.4f}, F1: {train_f1s[-1]:.4f}'
        )

        # Evaluation on Dev set
        evaluate(model, criterion, dev_dataloader, device=device, cm=False, cr=False, last_epoch=epoch == epochs-1)

        if saving_path:
            torch.save(model.state_dict(), saving_path)
            print('Saved Model!')