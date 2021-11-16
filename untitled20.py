from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch


def load_file(data_file_path):

    x_data, y_data = [], []

    # Get data from iob file
    if data_file_path.endswith('.iob') or data_file_path.endswith('.tsv'):
        x_data, y_data = load_iob(data_file_path)

    return x_data, y_data


def load_iob(file_path, extra_features=False):

    sents, labels = [], []
    words, tags = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if "DOCSTART" in line:
                continue
            line = line.rstrip()
            if line:
                cols = line.split('\t')
                if extra_features:
                    words.append([cols[0]] + cols[3:-1])
                else:
                    words.append(cols[0])
                tags.append(cols[-1])
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
        return sents, labels


def transform_labels(labels_list):
    new_labels = []
    for label in labels_list:
        if label == 'O':
            new_labels.append(label)
        elif '#' in label:
            new_labels.append(label.split('#')[-1])
        else:
            new_labels.append(label)
    return new_labels


class NerDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def main():
    model = AutoModelForTokenClassification.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
    x, y = load_file('nkjp-nested-simplified-v2.iob')
    y_cut = [transform_labels(labels) for labels in y]
    y_tokenized = [tokenizer(sentence, padding='max_length') for sentence in y_cut]
    x_tokenized = [tokenizer(label, padding='max_length') for label in x]

    optimizer = AdamW(model.parameters(), lr=5e-5)
    x_train, x_rem, y_train, y_rem = train_test_split(x_tokenized, y_tokenized, test_size=0.2, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

    train_dataset = NerDataset(x_train, y_train)
    val_dataset = NerDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

main()