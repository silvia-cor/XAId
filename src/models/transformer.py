from __future__ import annotations

from tqdm import tqdm

import numpy
import pandas
import random
import torch.cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Transformer:
    def __init__(self, model: str = "pstroe/roberta-base-latin-cased3", state_dict=None, num_labels=2, seed=42):
        # for reproducibility
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        numpy.random.seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.batch_size = 16
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(torch.load(state_dict))

    def fit(self, data: pandas.DataFrame, task: str, **hyperparameters):
        # Data
        dataloader = AuthorshipDataloader(data, task, self.tokenizer, self.batch_size, shuffle=True).dataloader

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        epochs = hyperparameters.get("epochs", 5)

        for epoch in range(epochs):
            epoch_loss = []
            with tqdm(dataloader, unit="batch") as train:
                self.model.train()
                for _, inputs, segs, masks, labels in train:
                    inputs = inputs.to(self.device)
                    segs = segs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    preds = self.model(inputs, token_type_ids=segs, attention_mask=masks).logits
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    train.set_description(f'Epoch {epoch + 1} loss={numpy.mean(epoch_loss):.5f}')

    def encode(self, data, task):
        dataloader = AuthorshipDataloader(data, task, self.tokenizer, self.batch_size, shuffle=False).dataloader
        encod_data = []
        with torch.no_grad():
            with tqdm(dataloader, unit="batch") as data:
                self.model.eval()
                for _, inputs, segs, masks, _ in data:
                    inputs = inputs.to(self.device)
                    segs = segs.to(self.device)
                    masks = masks.to(self.device)
                    output = self.model(inputs, token_type_ids=segs, attention_mask=masks, output_hidden_states=True)
                    encod_data.extend(
                        output.hidden_states[-1].mean(axis=1).detach().cpu().numpy())  # mean embeddings (N x 768)
        return encod_data

    def predict(self, test_df, task):
        predicted_labels = []
        test_data = AuthorshipDataloader(test_df, task, self.tokenizer, self.batch_size, shuffle=False).dataloader
        with torch.no_grad():
            with tqdm(test_data, unit="batch") as test:
                self.model.eval()
                for _, inputs, segs, masks, _ in test:
                    inputs = inputs.to(self.device)
                    segs = segs.to(self.device)
                    masks = masks.to(self.device)
                    preds = self.model(inputs, token_type_ids=segs, attention_mask=masks).logits
                    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                    predicted_labels.extend(preds)
        return predicted_labels


class AuthorshipDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class AuthorshipDataloader:
    def __init__(self, data: pandas.DataFrame, task: str, tokenizer: AutoTokenizer, batch_size: int, shuffle=False):
        self.task_data = [(data.text_A.values[i], data.text_B.values[i]) for i in range(len(data.text_A.values))] \
            if task == 'sav' else data.text.values
        self.labels = data.label.values
        self.tokenizer = tokenizer
        self.task = task
        dataset = AuthorshipDataset(self.task_data, self.labels)
        self.dataloader = DataLoader(dataset, batch_size, num_workers=5, shuffle=shuffle,
                                     worker_init_fn=self._seed_worker, collate_fn=self._collate_fn)

    # set the seed for the DataLoader worker
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def _collate_fn(self, batch):
        documents = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs, segs, masks = list(), list(), list()
        if self.task == "sav":
            for pair in documents:
                encoded_a = self.tokenizer.encode(pair[0], max_length=250, truncation=True, add_special_tokens=False)
                encoded_b = self.tokenizer.encode(pair[1], max_length=250, truncation=True, add_special_tokens=False)
                encoded_pair = torch.tensor([self.tokenizer.cls_token_id] + encoded_a + [self.tokenizer.sep_token_id]
                                            + encoded_b + [self.tokenizer.sep_token_id])
                length_a = len(encoded_a)
                length_b = len(encoded_b)
                segmentation = torch.tensor([0] * (length_a + 2) + [1] * (length_b + 1))
                mask = torch.tensor([1] * (length_a + length_b + 3))
                inputs.append(encoded_pair)
                segs.append(segmentation)
                masks.append(mask)
        else:
            for text in documents:
                encoded_text = self.tokenizer.encode(text, max_length=500, truncation=True, add_special_tokens=False)
                encoded_text = torch.tensor(
                    [self.tokenizer.cls_token_id] + encoded_text + [self.tokenizer.sep_token_id])
                length = len(encoded_text)
                segmentation = torch.tensor([0] * length)
                mask = torch.tensor([1] * length)
                inputs.append(encoded_text)
                segs.append(segmentation)
                masks.append(mask)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        segs = pad_sequence(segs, batch_first=True, padding_value=1) if self.task == 'sav' else \
            pad_sequence(segs, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return documents, inputs, segs, masks, torch.tensor(labels)
