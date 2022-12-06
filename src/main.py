import argparse

from transformers import AutoTokenizer

from src.constants import SENTIMENT_MAPPING
from src.data_processor import DataProcessor
from src.model_params import TEST_SIZE
from src.train_model import TrainingModel


def main():
    parser = argparse.ArgumentParser(description="Training a classifier for text sentiment prediction")
    parser.add_argument("--data_path", type=str,
                        help="Path to .csv file, containing tweets (and their sentiment- optional)")
    parser.add_argument("--label_col", type=str, help="Name of label column in the file")
    parser.add_argument("--text_col", type=str, help="Name of text column in the file")
    parser.add_argument("--model_dir", type=str,
                        help="Path to directory in which the trained model should be saved")
    parser.add_argument("--lang_model", type=str, default="bert-base-uncased",
                        help="Type of language model to use for pretrained model and tokenization")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.lang_model, do_lower_case=True)
    data_processor = DataProcessor(tokenizer)
    train_loader, val_loader = data_processor.process_texts_from_file(args.data_path,
                                                                      args.text_col,
                                                                      args.label_col,
                                                                      test_size=TEST_SIZE)
    model = TrainingModel(args.lang_model, num_labels=len(SENTIMENT_MAPPING))
    model.train(train_loader, val_loader, args.model_dir)


if __name__ == '__main__':
    main()
