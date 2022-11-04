from __future__ import annotations

from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy
import pandas
import torch.cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from torch.utils.data import Dataset


class TransformerTrainer:
    def __init__(self, model: str = "khosseini/bert_1760_1900"):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = None

    def fit(self, data: pandas.DataFrame, validation_data: pandas.DataFrame, task: str, **hyperparameters) -> AutoModelForSequenceClassification:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if task == "aa":
            num_labels = numpy.unique(data.label.values).size
        else:
            num_labels = 2
        # TODO: Remove
        return BertModel.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
        # TODOEND

        # Data
        data_loader = VictoriaLoader(data, task, self.tokenizer, device)
        validation_data_loader = VictoriaLoader(validation_data, task, self.tokenizer, device)
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        validation_metric = f1_score
        max_epochs = hyperparameters.get("max_epochs", 50)
        patience = hyperparameters.get("patience", 3)

        validation_scores, epochs_loss = list(), list()
        validation_score, validation_score_max = 0, 0
        patience_level = 0
        for epoch in range(max_epochs):
            epoch_loss = []
            with tqdm(data_loader, unit="batch") as train:
                self.model.train()
                for inputs, segmentation, mask, label in train:
                    optimizer.zero_grad()
                    label_predicted = self.model(inputs, token_type_ids=segmentation, attention_mask=mask, labels=label).logits
                    loss = criterion(label_predicted, label)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                epochs_loss.append(numpy.mean(epoch_loss))

                validation_labels = torch.Tensor([self.model(inputs, token_type_ids=segmentation, attention_mask=mask,
                                                             labels=targets).logits
                                                  for inputs, segmentation, mask, targets in validation_data_loader])
                validation_labels = torch.argmax(validation_labels, dim=1)
                validation_score = validation_metric(validation_data, validation_labels, average="binary")

                if validation_score > validation_score_max:
                    validation_score_max = validation_score
                    patience_level = 0
                else:
                    if epoch >= 4:
                        patience_level += 1
                validation_scores.append(validation_score)
                if patience_level == patience:
                    break

        return self.model


class AuthorshipDataLoader(Dataset):
    def __init__(self):
        self.documents, self.task, self.labels = [], [], []


class VictoriaLoader(AuthorshipDataLoader):
    """
    Args:
        texts: List[str]: Texts in this DataLoader
        encoded_texts: torch.Tensor: The encoded texts
        task: str: The task for which this DataLoader has been created
    """
    def __init__(self, data: pandas.DataFrame, task: str, tokenizer: AutoTokenizer, device: str):
        super().__init__()
        self.documents, self.encoded_texts, self.segmentations, self.masks = list(), list(), list(), list()
        self.labels = data.label.values
        self.task = task

        for idx, row in data.iterrows():
            if task == "sav":
                encoded_a = tokenizer.encode(row["text_A"], max_length=254, truncation=True, add_special_tokens=False)
                encoded_b = tokenizer.encode(row["text_B"], max_length=254, truncation=True, add_special_tokens=False)
                encoded_pair = torch.tensor([tokenizer.cls_token_id] + encoded_a + [tokenizer.sep_token_id]
                                            + encoded_b + [tokenizer.sep_token_id]).to(device)
                length_a = len(encoded_a)
                length_b = len(encoded_b)
                segmentation = torch.tensor([0] * (length_a + 2) + [1] * (length_b + 1)).to(device)
                mask = torch.tensor([1] * (length_a + length_b + 3)).to(device)

                self.documents.append(row["text_A"] + " [SEP] " + row["text_B"])
                self.encoded_texts.append(encoded_pair)
                self.segmentations.append(segmentation)
                self.masks.append(mask)
            elif task == "av":
                encoded_text = tokenizer.encode(row["text"], max_length=254, truncation=True, add_special_tokens=False).to(device)
                encoded_author = tokenizer.encode(str(row["author"]), truncation=True, add_special_tokens=False)
                encoded_pair = torch.tensor([tokenizer.cls_token_id] + encoded_text + [tokenizer.sep_token_id]
                                            + encoded_author + [tokenizer.en]).to(device)

                # self.encoded_texts.append(encoded_pair)
                length_text = len(encoded_text)
                length_author = len(encoded_author)
                segmentation = torch.tensor([0] * (length_text + 2) + [1] * (length_author + 1)).to(device)
                mask = torch.tensor([1] * (length_text + length_author + 3)).to(device)

                self.documents.append(row["text"])
                self.encoded_texts.append(encoded_pair)
                self.segmentations.append(segmentation)
                self.masks.append(mask)
            elif task == "aa":
                encoded_text = tokenizer.encode(row["text"], max_length=254, truncation=True,
                                                add_special_tokens=False)
                encoded_text = torch.tensor([tokenizer.cls_token_id] + encoded_text + [tokenizer.sep_token_id]).to(device)

                length = len(encoded_text)
                segmentation = torch.tensor([0] * (length + 2)).to(device)
                mask = torch.tensor([1] * (length + 2)).to(device)

                self.documents.append(row["text"])
                self.encoded_texts.append(encoded_text)
                self.segmentations.append(segmentation)
                self.masks.append(mask)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, item: int):
        return self.documents[item], self.encoded_texts[item], self.segmentations[item], self.masks[item], self.labels[item]
