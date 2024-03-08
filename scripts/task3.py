import torch 
import pandas as pd
import numpy as np
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import random_split
from task3_data_preparation import prepare_data

MODEL_NAME = "facebook/xglm-564M"

########################################################
# Entry point
########################################################

if __name__ == "__main__":
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
    train_dataset_list, evaluation_dataset_list = prepare_data(dataset_path, data_files)
    

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
    
    print(len(train_dataset_list))
    # tokenize the sentences
    tokenized_train_datasets = [tokenization(sentence) for sentence in train_dataset_list]
    tokenized_eval_datasets = [tokenization(sentence) for sentence in evaluation_dataset_list]
    print(len(tokenized_train_datasets))
    print("yayy")


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
    xglm_trainer = Trainer(xglm_model, training_args, data_collator=data_collator, train_dataset=tokenized_train_datasets, eval_dataset=tokenized_eval_datasets, tokenizer=xglm_tokenizer)

    for i in tqdm.tqdm(range(10000), desc='Processing'):
        # start training
        xglm_trainer.train()
