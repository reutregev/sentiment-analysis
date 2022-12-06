from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BatchEncoding, AutoTokenizer

from src.constants import LABEL, INPUT_IDS, ATTENTION_MASK, SENTIMENT_MAPPING
from src.logger import logger
from src.model_params import BATCH_SIZE


class DataProcessor:
    """
    Class for processing data before passing it to the model, including:
    reading, splitting to train and validation (optional), tokenizing, and storing in dataloader.
    The data can be either read from a CSV file or received in a list of text.
    """

    def __init__(self, tokenizer: AutoTokenizer, batch_size: int = BATCH_SIZE):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @staticmethod
    def load_from_csv(data_path: str,
                      text_col: str,
                      label_col: str,
                      labels_mapping: Dict = SENTIMENT_MAPPING
                      ) -> pd.DataFrame:
        """
        Read data from .csv file that includes texts under text_col and sentiment under label_col.
        Then, rename the label_col and map the label values to their ids.

        Parameters
        ----------
        data_path : str
            Path to .csv file
        text_col : str
            Name of text column in the file
        label_col : str
            Name of label column
        labels_mapping : Dict[str, int]
            Mapping between sentiment label and integer label

        Returns
        -------
        DataFrame

        """
        logger.info(f"Reading data from file {data_path}")
        data = pd.read_csv(data_path)[[text_col, label_col]]
        # normalize the column name and the labels
        if label_col != LABEL:
            data = data.rename({label_col: LABEL}, axis=1)
        data[LABEL] = data[LABEL].map(labels_mapping)

        return data

    def encode_text(self, texts: List[str]) -> BatchEncoding:
        """ Tokenize given texts and return it as BatchEncoding """
        logger.info("Encoding texts")
        return self.tokenizer.batch_encode_plus(texts,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')

    def process_texts_from_file(self,
                                data_path: str,
                                text_col: str,
                                label_col: str,
                                labels_mapping: Dict = SENTIMENT_MAPPING,
                                test_size: float = None
                                ) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
        """
        Process texts of a single CSV file by performing tokenization and creating a dataloader
        If test_size is passed, a tuple of 2 dataloaders is returned (train and validation).
        If not, no split is done and a single dataloader is returned (for testing purposes)

        Parameters
        ----------
        data_path : str
            Path to csv file
        text_col : str
            Name of text column as appears in the file
        label_col : str
            Name of label column as appears in the file
        labels_mapping : Dict[str, int]
            Mapping between label strings to integers
        test_size : float, optional
            Fraction of test set size

        Returns
        -------
        Union[Tuple[DataLoader, DataLoader], DataLoader]
            Tuple of DataLoader for training and validation or
            a single DataLoader for test

        """
        data_df = self.load_from_csv(data_path, text_col, label_col, labels_mapping)
        if test_size:
            X_train, X_val, y_train, y_val = train_test_split(data_df[text_col].values,
                                                              data_df[LABEL].values,
                                                              test_size=test_size,
                                                              random_state=27,
                                                              stratify=data_df[LABEL].values)
            train_loader = self.process_texts(X_train.tolist(), y_train.tolist())
            val_loader = self.process_texts(X_val.tolist(), y_val.tolist())
            return train_loader, val_loader
        else:
            data_df = pd.read_csv(data_path)
            return self.process_texts(data_df[text_col].tolist())

    def process_texts(self, texts: List[str], data_labels: List[int] = None) -> DataLoader:
        """
        Process a list of text: tokenize and store in DataLoader.

        Parameters
        ----------
        texts : List[str]
        data_labels: List[int], optional
            List of text labels

        Returns
        -------
        DataLoader

        """
        logger.info(f"Processing texts")
        text_encodings = self.encode_text(texts)
        if data_labels:
            return self.create_dataloader(text_encodings, self.batch_size, data_labels)
        return self.create_dataloader(text_encodings, self.batch_size)

    @staticmethod
    def create_dataloader(text_encoded: BatchEncoding,
                          batch_size: int = BATCH_SIZE,
                          data_labels: List[int] = None
                          ) -> DataLoader:
        """
        Create a dataloader for the encoded text and labels.
        Pass data_labels to create a dataloader for training purposes. If data_labels are not passes, the dataloader
        could be used for test purposes.

        Parameters
        ----------
        text_encoded : BatchEncoding
            Texts encoded by tokenizer
        batch_size : int
            Batch size
        data_labels: List[int], optional
            List of text labels

        Returns
        -------
        DataLoader

        """
        logger.info("Creating dataloaders")
        if data_labels:
            labels_tensor = torch.tensor(data_labels)
            data_tensor = TensorDataset(text_encoded[INPUT_IDS], text_encoded[ATTENTION_MASK], labels_tensor)
        else:
            data_tensor = TensorDataset(text_encoded[INPUT_IDS], text_encoded[ATTENTION_MASK])
        sampler = RandomSampler(data_tensor) if data_labels else None
        return DataLoader(data_tensor, batch_size=batch_size, sampler=sampler)
