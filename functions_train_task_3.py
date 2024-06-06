import torch
from tqdm import tqdm


def train(model, optimizer, tokenizer, epochs, train_dataloader, saving_path=None, device='cpu'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        

        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

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

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}')

        if saving_path:
            path = saving_path + "_" + str(epoch) + '.pth'
            torch.save(model.state_dict(), path)
            print('Saved Model in ' +  path)


            