from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification
from general.process_data import SAV_DataLoader, AA_AV_test_DataLoader
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import f1_score
from torch import nn
import os

device = torch.device(3 if torch.cuda.is_available() else 'cpu')

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def NN_experiment(tr_data, val_data, te_data, task, model_path, AV_label, unique_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Creating dataloaders...')
    tr_dataloader = SAV_DataLoader(tr_data, tokenizer, batch_size=16)
    print('Train dataloader: Done')
    val_dataloader = SAV_DataLoader(val_data, tokenizer, batch_size=16)
    print('Val dataloader: Done')
    if task == 'SAV':
        te_dataloader = SAV_DataLoader(te_data, tokenizer, batch_size=16)
    else:
        te_dataloaders = AA_AV_test_DataLoader(te_data, tokenizer)
    print('Test dataloader: Done')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    max_epochs = 500
    patience = 5

    if os.path.exists(model_path):
        print('Pair classifier found!')
    else:
        epochs_loss, val_f1s = _train_NN(model, optimizer, tr_dataloader, val_dataloader, criterion, max_epochs,
                                         patience, model_path)
    if task == 'SAV':
        preds, targets = _evaluate_class(model, te_dataloader, model_path)
        return preds, targets
    else:
        preds = _evaluate_probs(model, te_dataloaders, model_path, AV_label, unique_labels)
        print(te_data['task_labels'])
        return preds, te_data['task_labels']


def _train_NN(model, optimizer, tr_dataloader, val_dataloader, criterion, max_epochs, patience, model_path):
    val_f1s, epochs_loss = [], []
    epochs_no_improv = 0
    val_f1, val_f1max = 0, 0
    for epoch in range(max_epochs):
        epoch_loss = []  # loss of single epoch (one for each batch)
        with tqdm(tr_dataloader, unit="batch") as train:
            model.train()
            tr_all_preds = []
            tr_all_targets = []
            for input_ids, mask_ids, seg_ids, targets in train:
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                targets = targets.to(device)
                preds = model(input_ids, attention_mask=mask_ids, token_type_ids=seg_ids, labels=targets).logits
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                preds = torch.argmax(preds, dim=1)
                tr_all_preds.extend(preds.detach().clone().cpu().numpy())
                tr_all_targets.extend(targets.detach().clone().cpu().numpy())
                train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
            tr_f1 = f1_score(tr_all_targets, tr_all_preds, average='binary')
            print(f'Tr_F1: {tr_f1:.3f}')
            epochs_loss.append(np.mean(epoch_loss))
            if epoch == 0:
                torch.save(model.state_dict(), model_path)
            val_all_preds, val_all_targets = _evaluate_class(model, val_dataloader, model_path)
            print(val_all_preds)
            print(val_all_targets)
            val_f1 = f1_score(val_all_targets, val_all_preds, average='binary')
            # if after patience there is no improvement, early stop happens
            # for the first 10 epochs, no early stoppings
            if val_f1 >= val_f1max:
                torch.save(model.state_dict(), model_path)
            if val_f1 > val_f1max:
                val_f1max = val_f1
                epochs_no_improv = 0
            else:
                if epoch > 10:
                    epochs_no_improv += 1
            val_f1s.append(val_f1)
            print(f'Val_F1max: {val_f1max:.3f} Val_F1: {val_f1:.3f}')
            if epochs_no_improv == patience and epoch > 9:
                print("Early stopping!")
                break
    print('Training on validation set')
    for epoch in range(10):
        epoch_loss = []
        with tqdm(val_dataloader, unit="batch") as val:
            model.train()
            for token_ids, mask_ids, seg_ids, targets in val:
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                targets = targets.to(device)
                preds = model(pair_token_ids, token_type_ids=seg_ids, attention_mask=mask_ids).logits
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
    torch.save(model.state_dict(), model_path)
    return epochs_loss, val_f1s


def _evaluate_class(model, dataloader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for input_ids, mask_ids, seg_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            preds = model(input_ids, token_type_ids=seg_ids, attention_mask=mask_ids).logits
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_targets.extend(targets.numpy())
    return all_preds, all_targets


def _evaluate_probs(model, dataloaders, model_path, AV_label, unique_labels):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        all_probs = []
        all_pairs_labels = []
        for dataloader in dataloaders:
            for token_ids, mask_ids, seg_ids, pairs_labels in dataloader:
                token_ids = token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                all_pairs_labels.append(pairs_labels)
                all_probs.append(model(token_ids, token_type_ids=seg_ids,
                                       attention_mask=mask_ids).logits.detach().clone().cpu().numpy())
        preds = []
        for i, probs in enumerate(all_probs):
            label_probs = []  # mean probability that text share same author with each label
            for label in unique_labels:
                label_probs.append(np.mean(np.array([pair_probs[1] for j, pair_probs in enumerate(probs)
                                                     if all_pairs_labels == label])))
            preds.append(unique_labels[np.argmax(np.array(label_probs))])
        print(len(preds), preds)
        if AV_label:
            preds = [1 if single_y_pred == AV_label else 0 for single_y_pred in preds]
            print(preds)
    return preds
