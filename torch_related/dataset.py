import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['Review'].to_numpy()
        self.labels = dataframe['Rating'].to_numpy()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure labels are tensors
        }
