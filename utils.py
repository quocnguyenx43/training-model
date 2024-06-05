import numpy as np


def my_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def my_recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def my_hamming_loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])


def my_precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def my_f1_score(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]


def vit5_encode(source_text, target_text, source_len, target_len, tokenizer):
    source_encoded = tokenizer.batch_encode_plus(
        source_text,
        max_length= source_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    source_encoded['input_ids'] = source_encoded['input_ids'].squeeze()
    source_encoded['attention_mask'] = source_encoded['attention_mask'].squeeze()

    target_encoded = tokenizer.batch_encode_plus(
        target_text,
        max_length= target_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    target_encoded['input_ids'] = target_encoded['input_ids'].squeeze()
    target_encoded['attention_mask'] = target_encoded['attention_mask'].squeeze()

    return source_encoded, target_encoded