import random
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
from process_data import SavDataLoader, AaAvTestDataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def nn_experiment(tr_data, val_data, te_data, task, model_path, unique_labels):
    """
    Manage the Neural Network experiment.
    :param tr_data: training data
    :param val_data: validation data
    :param te_data: test data
    :param task: task to perform
    :param model_path: file path where to save the SAV model
    :param unique_labels: list of unique labels
    :return: predictions and targets
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Creating dataloaders...')
    tr_dataloader = SavDataLoader(tr_data, tokenizer, batch_size=16)
    print('Train dataloader: Done')
    val_dataloader = SavDataLoader(val_data, tokenizer, batch_size=16)
    print('Val dataloader: Done')
    if task == 'SAV':
        te_dataloader = SavDataLoader(te_data, tokenizer, batch_size=16)
    else:
        te_dataloaders = AaAvTestDataLoader(te_data, tokenizer, task)
    print('Test dataloader: Done')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    max_epochs = 50  # maximum number of epochs for training
    patience = 3  # number of epoch without improvement before early stopping

    if os.path.exists(model_path):
        print('Pair classifier found!')
    else:
        epochs_loss, val_f1s = _train_nn(model, optimizer, tr_dataloader, val_dataloader, criterion, max_epochs,
                                         patience, model_path)
    if task == 'SAV':
        preds, targets = _evaluate_class(model, te_dataloader, model_path)
        return preds, targets
    else:
        preds = _evaluate_probs(model, te_dataloaders, model_path, task, unique_labels)
        return preds, te_data['task_labels']


# training the SAV model
def _train_nn(model, optimizer, tr_dataloader, val_dataloader, criterion, max_epochs, patience, model_path):
    val_f1s, epochs_loss = [], []  # list of f1s obtained in validation | list of losses obtained in training (for plotting)
    epochs_no_improv = 0  # epochs passed without improvements
    val_f1, val_f1max = 0, 0  # f1 obtained in validation for current epoch | max f1 obtained in validation
    for epoch in range(max_epochs):
        epoch_loss = []  # loss of single epoch (one for each batch)
        with tqdm(tr_dataloader, unit="batch") as train:
            model.train()
            tr_all_preds = []
            tr_all_targets = []
            for input_ids, seg_ids, mask_ids, targets in train:
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                seg_ids = seg_ids.to(device)
                mask_ids = mask_ids.to(device)
                targets = targets.to(device)
                preds = model(input_ids, token_type_ids=seg_ids, attention_mask=mask_ids, labels=targets).logits
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
            val_f1 = f1_score(val_all_targets, val_all_preds, average='binary')
            # if after patience there is no improvement, early stop happens
            # for the first 5 epochs, no early stoppings
            if val_f1 >= val_f1max:
                torch.save(model.state_dict(), model_path)
            if val_f1 > val_f1max:
                val_f1max = val_f1
                epochs_no_improv = 0
            else:
                if epoch >= 4:
                    epochs_no_improv += 1
            val_f1s.append(val_f1)
            print(f'Val_F1max: {val_f1max:.3f} Val_F1: {val_f1:.3f}')
            if epochs_no_improv == patience:
                print("Early stopping!")
                break
    print('Training on validation set')  # final training on validation set
    for epoch in range(5):
        epoch_loss = []
        with tqdm(val_dataloader, unit="batch") as val:
            model.train()
            for input_ids, seg_ids, mask_ids, targets in val:
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                seg_ids = seg_ids.to(device)
                mask_ids = mask_ids.to(device)
                targets = targets.to(device)
                preds = model(input_ids, token_type_ids=seg_ids, attention_mask=mask_ids).logits
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                val.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
    torch.save(model.state_dict(), model_path)
    return epochs_loss, val_f1s


# evaluation outputting classes directly (for SAV)
def _evaluate_class(model, dataloader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for input_ids, seg_ids, mask_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            seg_ids = seg_ids.to(device)
            mask_ids = mask_ids.to(device)
            preds = model(input_ids, token_type_ids=seg_ids, attention_mask=mask_ids).logits
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_targets.extend(targets.numpy())
    return all_preds, all_targets


# evaluation outputting classes through probabilities (for AA and AV)
def _evaluate_probs(model, dataloaders, model_path, task, unique_labels):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        all_probs = []
        preds = []
        if task == 'AA':
            all_pairs_labels = []
            for dataloader in dataloaders:
                for token_ids, seg_ids, mask_ids, pairs_labels in dataloader:
                    token_ids = token_ids.to(device)
                    seg_ids = seg_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    all_pairs_labels.append(pairs_labels)
                    all_probs.append(model(token_ids, token_type_ids=seg_ids,
                                           attention_mask=mask_ids).logits.detach().clone().cpu().numpy())
            for i, single_test_probs in enumerate(all_probs):
                label_probs = []  # mean probability that the test sample belongs to each label
                for label in unique_labels:
                    label_probs.append(np.mean(np.array([pair_probs[1] for j, pair_probs in enumerate(single_test_probs)
                                                         if all_pairs_labels[i][j] == label])))
                preds.append(unique_labels[np.argmax(np.array(label_probs))])
        else:
            for dataloader in dataloaders:
                for token_ids, seg_ids, mask_ids, in dataloader:
                    token_ids = token_ids.to(device)
                    seg_ids = seg_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    all_probs.append(model(token_ids, token_type_ids=seg_ids,
                                           attention_mask=mask_ids).logits.detach().clone().cpu().numpy())
            for i, single_test_probs in enumerate(all_probs):
                # for AV, check if the pairs with the author of interest have a mean probability >= 0.5
                preds.append(1 if np.mean(np.array([pair_probs[1] for pair_probs in single_test_probs])) >= 0.5 else 0)
    return preds
