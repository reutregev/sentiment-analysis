import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.logger import logger
from src.utils import calc_running_time


class InferenceModel:

    def __init__(self, model_path: str, lang_model: str = "bert-base-uncased"):
        """
        Instantiate a model for inference.

        Parameters
        ----------
        model_path : str
            Path to trained model
        lang_model : str, optional
            Type of language model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model, do_lower_case=True)
        logger.info(f"Loaded a trained model from {model_path}")

    @calc_running_time
    def infer(self, inference_dataloader: DataLoader):
        """
        Run inference on the given texts

        Parameters
        ----------
        inference_dataloader : DataLoader
            DataLoader for inference

        Returns
        -------
        float, ndarray
            Inference loss, predictions
        """
        logger.info("Running inference")
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for input_ids, attention_mask in inference_dataloader:
                logits = self.model(input_ids, attention_mask).logits
                all_preds.extend(logits.argmax(dim=1).numpy().tolist())

        return all_preds
