# column names
TEXT = "text"
LABEL = "label"

# sentiment mapping dict
SENTIMENT_MAPPING = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}

# label to sentiment mapping dict
LABEL_MAPPING = {
    v: k for k, v in SENTIMENT_MAPPING.items()
}

# BatchEncoding keys
INPUT_IDS = "input_ids"
TOKEN_TYPE_IDS = "token_type_ids"
ATTENTION_MASK = "attention_mask"
