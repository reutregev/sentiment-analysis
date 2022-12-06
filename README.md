# Text Sentiment Analysis

This project utilizes a pre-trained language model (PyTorch) for the downstream task of sentiment classification of text.<br>

The trained classifier is served via REST API built with Flask.<br>

## How to use
### Training

The dataset used for training is [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment),
which includes reviews on US Airline companies and their sentiment (positive, negative, or neutral).<br>
The data is split into train and validation, which are preprocessed by using a tokenizer that corresponds to the language model.<br>

To build the classifier, the type of language model should be passed as an input, otherwise it uses base BERT as a default.<br>
The added model files (as git LFS artifacts) are of trained classifier based on pre-trained `bert-base-uncased`.<br>

Below is an example to run the training:

```commandline
python ./src/main.py [ARGUMENTS] [OPTIONS]

Arguments:
    --data_path     Path to .csv file, containing tweets (and their sentiment- optional)
    --label_col     Name of label column in the file
    --text_col      Name of text column in the file
    --model_dir     Path to directory in which the trained model should be saved

Options:
    --lang_model    Type of language model to use for pretrained model and tokenization
```

### Inference
For inference, you should run either the Flask app or the docker: <br>
- Flask:
```commandline
python ./app/app.py [OPTIONS]

Options:
    --model_dir     Path to directory where the trained model is saved
    --lang_model    Type of language model to use for tokenization. Must correspond to the one used in the training phase
```

- Docker:
  - Build the docker image:
    ```commandline
    docker build -t <docker_name> -f ./docker/Dockerfile .
    ```
  - Once the image is built, run the container:
    ```commandline
    docker run -d -p 9980:9980 <docker_name>
    ```
Once the service is running, run the following request to get sentiment predictions:
```commandline
    curl --location --request POST 'http://127.0.0.1:9980/predict' \
         --header 'Content-Type: application/json' \
         --data-raw '{
                        "text1": "Had a terrible experience flying with you!",
                        "text2": "The best airline company ever!!!"
                     }'
```
Where `--data-raw` contains the texts for prediction in json format.





