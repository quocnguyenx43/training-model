import torch



def train(model, optimizer, tokenizer, train_dataloader, saving_path=None, device='cpu'):

    for epoch in range(5):
        model.train()
        running_loss = 0.0

        print('epoch', epoch)
        
        for _, batch in enumerate(train_dataloader, 0):
            print('batch', batch)
            ids = batch['input']['input_ids'].to(device, dtype=torch.long)
            mask = batch['input']['attention_mask'].to(device, dtype=torch.long)

            y = batch['label']['input_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            running_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if saving_path:
            pass