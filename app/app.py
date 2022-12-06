import argparse
import os

from flask import Flask, request
from transformers import AutoTokenizer

from src.constants import LABEL_MAPPING
from src.data_processor import DataProcessor
from src.infer_model import InferenceModel
from src.logger import logger

app = Flask(__name__)
VERSION = os.getenv('VERSION', '1.0.0')


@app.route("/", methods=["GET"])
def home():
    return "Home"


@app.route("/version", methods=["GET"])
def version():
    return VERSION


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a text and predict its sentiment

    Usage example:
    curl --location --request POST 'http://127.0.0.1:9980/predict'
         --header 'Content-Type: application/json'
         --data-raw '{"text1": "Had a terrible experience flying with you!",
                      "text2": "The best airline company ever!!!"}'
    """
    texts = list(request.json.values())
    text_data_loader = data_processor.process_texts(texts)
    preds = trained_model.infer(text_data_loader)

    for text, pred in zip(texts, preds):
        logger.info(f"Text: `{text}` | Sentiment: {LABEL_MAPPING[pred]}")

    return dict(zip(texts, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Application for text sentiment prediction")
    parser.add_argument("--model_dir", type=str, default="model/trained_model.pt",
                        help="Path to directory where the trained model is saved")
    parser.add_argument("--lang_model", type=str, default="bert-base-uncased",
                        help="Type of language model to use for tokenization. Must correspond to the one used in the training phase")
    args = parser.parse_args()

    trained_model = InferenceModel(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.lang_model, do_lower_case=True)
    data_processor = DataProcessor(tokenizer)

    app.run(debug=True, host="0.0.0.0", port=9980)
