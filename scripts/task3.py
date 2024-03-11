import torch 
import pandas as pd
import numpy as np
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import random_split
from task3_data_preparation import prepare_data_qu, prepare_data_fine_tune_eval
import evaluate

MODEL_NAME = "facebook/xglm-564M"

########################################################
# Entry point
########################################################

if __name__ == "_main_":
    # TODO: your code goes here

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Steps:
    # 1) Prepare the dataset
    # 
    # 2) Perform training => use class transformers.Trainer


    # Step 1)

    # Load the dataset
    dataset_path = "hackathon-pln-es/spanish-to-quechua"

    data_files = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
    }
    # train_dataset_list and evaluation_dataset_list contain sentences in "qu" as a list
    train_dataset_list, evaluation_dataset_list = prepare_data_qu(dataset_path, data_files)

    # specify languages that will be used to evaluate the model's performance
    # during fine-tuning.
    LANGUAGES = [
        "eng_Latn",
        "spa_Latn",
        "ita_Latn",
        "deu_Latn",
        "arb_Arab",
        "tel_Telu",
        "tam_Taml"
    ]

    # while fine_tuning, we need to check the model's performance on other 
    # languages as well to see how the performance gets affected.
    prepare_data_fine_tune_eval = prepare_data_fine_tune_eval(LANGUAGES)

    # append the dataset obtained for fine_tuning to the evaluation dataset
    prepare_data_fine_tune_eval["qu"] = evaluation_dataset_list
    

    # Step 2)

    # load pre-trained model from the huggingface hub
    xglm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # load a pre-trained tokenizer from the huggingface hub
    xglm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # since we are using a decoder only architecture, we need to perform the padding
    # on the left side so that the model does not get confused while guessing the next token.
    xglm_tokenizer.padding_side = 'left'

    # specify the tokenization function
    def tokenization(example):
        # fill in here
        tokenized = xglm_tokenizer(example, padding=True, truncation=True)
        return tokenized
    
    
    # tokenize the sentences in the train dataset, which is made up of only "qu" language
    tokenized_train_datasets = [tokenization(sentence) for sentence in train_dataset_list]

    # tokenize the sentences in the eval dataset, which is made of several languages. 
    tokenized_eval_datasets = {}
    for lang in prepare_data_fine_tune_eval.keys():
        tokenized_eval_datasets[lang] = [tokenization(sentence) for sentence in prepare_data_fine_tune_eval[lang]]

    # Setup evaluation 
    metric = evaluate.load("perplexity", module_type="metric")

    def compute_metrics(eval_pred):
        """
        This function will be used for model evaluation. 

        args:
            eval_pred: the output of the model. Made up of logits and labels, where logits
            are unnormalized scores that the model predicts. These logits are typically 
            transformed into probabilities using a softmax function. 

        returns:
            - Dictionary: the keys represent different sub-datasets and the evaluation on them separately.
        """

        # In evaluation mode, the output of an autoregressive model like XGLM consists of logits 
        # corresponding to the score for each possible next token in the vocabulary given the previous sequence of tokens. 
        # The label is the ground truth—the actual token that comes next in the sequence according to the dataset, 
        # regardless of the model's prediction. It's the target that the model is trying to predict.
        logits, labels = eval_pred

        # Convert logits to probabilities for all possible next words
        probabilities = torch.softmax(logits, axis=-1)
        
        # You need the probabilities of the actual next tokens, referenced by the labels
        true_next_token_probabilities = np.take_along_axis(probabilities, np.expand_dims(labels, -1), axis=-1)

        # Perplexity is calculated using the probabilities of the true next tokens
        perplexity = metric.compute(predictions=true_next_token_probabilities.squeeze(), references=labels)
        
        return perplexity


    

    # specify the training arguments. note that these are hyperparameters.
    # train_batch_size   
    per_device_train_batch_size = 6
    per_device_eval_batch_size = 6
    num_train_epochs = 1

    # data collators will form batches by using a list of dataset elements as input.
    # to be able to build batches, they may apply some processing (like padding).
    # since we are using causal language modeling, predict the next token in the sequence given the previous tokens,
    # we choose mlm to be false. 
    data_collator = DataCollatorForLanguageModeling(xglm_tokenizer, mlm=False)
    training_args = TrainingArguments(per_device_train_batch_size=per_device_train_batch_size, per_device_eval_batch_size=per_device_eval_batch_size, output_dir="trainer_outout", num_train_epochs=num_train_epochs)

    # we need to pass the model to the trainer (training and eval loop for PyTorch)
    # arguments: the model, train_dataset and eval_dataset should be torch.utils.data.Dataset or torch.utils.data.IterableDataset
    xglm_trainer = Trainer(xglm_model, training_args, data_collator=data_collator, train_dataset=tokenized_train_datasets, eval_dataset=tokenized_eval_datasets, tokenizer=xglm_tokenizer, compute_metrics=compute_metrics)

    for i in tqdm.tqdm(range(10000), desc='Processing'):
        # start training
        xglm_trainer.train()