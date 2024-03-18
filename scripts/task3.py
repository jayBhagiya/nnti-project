import os
import tqdm
import torch
import wandb
import evaluate
import datasets
import numpy as np

from task3_data_preparation import prepare_data_qu, prepare_data_fine_tune_eval
from task3_custom_peft import BitFitAdaptedModel, LoRaAdaptedModel, IA3AdaptedModel
from task3_utils import ParamsUtils

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

MODEL_NAME = "facebook/xglm-564M"
DATASET_NAME = "facebook/flores" # Just for the evalution of the languages defined in the task2

TRAIN_SAMPLES = 2000       # This number of samples from new dataset are used for fine-tuning the model
BATCH_SIZE = 2
MIN_EPOCHS = 10
MAX_EPOCHS = 100
GRAD_ACCUM = 4
GRAD_CHECKPOINTING = True
FP16 = True

PROJECT_NAME = "nnti-project"
ENTITY = "your-wandb-entity"

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

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

########################################################
# Functions
########################################################
def run_task3():
    ########################################################
    # Steps:
    # 1) Prepare the dataloaders
    # 2) Perform fine tuning
    # 3) Different Adaptation methods
    ########################################################

    #=======================================================
    # Step 1) Prepare the dataloaders
    #=======================================================

    # Load the dataset
    dataset_path = "hackathon-pln-es/spanish-to-quechua"

    data_files = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet"
    }

    # train_dataset_list and evaluation_dataset_list contain sentences in "qu" as a list
    train_dataset_qu, evaluation_dataset_qu = prepare_data_qu(dataset_path, data_files)

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
    dataset_eval = prepare_data_fine_tune_eval(LANGUAGES)

    # append the dataset obtained for fine_tuning to the evaluation dataset
    dataset_eval["que_Quec"] = evaluation_dataset_qu

    #=======================================================
    # 1:Prepare the dataloaders
    #   1.1: Dataloaders with the tokenized dataset.
    #=======================================================

    # load a pre-trained tokenizer from the huggingface hub
    xglm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator_pad = DataCollatorWithPadding(tokenizer=xglm_tokenizer)

    # since we are using a decoder only architecture, we need to perform the padding
    # on the left side so that the model does not get confused while guessing the next token.
    xglm_tokenizer.padding_side = 'left'

    # specify the tokenization function
    def tokenization(example):
        tokenized = xglm_tokenizer(example, padding=True, truncation=True)
        return tokenized

    # tokenize the sentences in the train dataset, which is made up of only "qu" language
    tokenized_train_datasets = [tokenization(sentence) for sentence in train_dataset_qu[:TRAIN_SAMPLES]]

    # tokenize the sentences in the eval dataset, which is made of several languages.
    tokenized_eval_datasets = {}
    for lang in dataset_eval.keys():
        tokenized_eval_datasets[lang] = [tokenization(sentence) for sentence in dataset_eval[lang][:TRAIN_SAMPLES]]

    train_dataloader = DataLoader(tokenized_train_datasets, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator_pad)

    # Construct a PyTorch DataLoader for each dataset
    eval_dataloaders = {
        lang: DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator_pad) for lang, dataset in tokenized_eval_datasets.items()
    }

    #=======================================================
    # Step 2) Perform fine tuning
    #=======================================================
    
    wandb.init(project=PROJECT_NAME, entity=ENTITY)
    
    #=======================================================
    # 2:Fine Tuning
    #   2.1: Models
    #=======================================================

    xglm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    bitfit_model = BitFitAdaptedModel(MODEL_NAME)
    lora_model = LoRaAdaptedModel(MODEL_NAME, rank=4, alpha=1)
    ia3_model = IA3AdaptedModel(MODEL_NAME)

    ParamsUtils(xglm_model, bitfit_model, "bitfit").print_stats()
    ParamsUtils(xglm_model, lora_model, "lora").print_stats()
    ParamsUtils(xglm_model, ia3_model, "ia3").print_stats()

    models = {
        "xglm-fine": xglm_model, 
        "bitfit": bitfit_model, 
        "lora": lora_model, 
        "ia3": ia3_model
    }

    optimizers = {
        name: AdamW(model.parameters(), lr=2e-4) for name, model in models.items()
    }

    for name, model in models.items():
        model.to(device)

    #=======================================================
    # 2:Fine Tuning
    #   2.2: Args setting
    #=======================================================

    num_epochs = MIN_EPOCHS
    num_training_steps = num_epochs * len(train_dataloader)
    
    lr_schedulers = {
        name: get_scheduler(
          "linear",
          optimizer=optimizers[name],
          num_warmup_steps=0,
          num_training_steps=num_training_steps
        ) for name, model in models.items()
    }

    #=======================================================
    # 2:Fine Tuning
    #   2.3: Training
    #=======================================================

    model_names = ["xglm-fine", "bitfit", "lora", "ia3"]

    for name in model_names:
        model = train(models[name], train_dataloader, num_epochs, optimizers[name], lr_schedulers[name], name)
        models[name] = model

        torch.save(models[name].state_dict(), f"{name}_model.pt")
        wandb.save(f"{name}_model.pt")

    #=======================================================
    # 2:Fine Tuning
    #   2.3: Evaluation
    #=======================================================

    for name in model_names:
        test_model(models[name], eval_dataloaders, name)

    wandb.finish()


def train(model, train_dataloader, num_epochs, optimizer, lr_scheduler, model_name):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, nn.CrossEntropyLoss(), log="all", log_freq=50)

    model.train()
    for epoch in tqdm.tqdm(range(num_epochs)):
        batch_step = 0
        loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone().to(device)
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            curr_loss = outputs.loss

            curr_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss += curr_loss
            batch_step += 1
            
            if ((batch_step + 1) % 50) == 0:
                wandb.log({f"qu-{model_name}-batch-loss": curr_loss}, step=batch_step)
                print(f"Loss after {str(batch_step).zfill(5)} batch examples: {curr_loss:.3f}")

        wandb.log({f"qu-{model_name}-avg-loss": loss / len(train_dataloader), "epoch": epoch})

    return model

def test_model(model, eval_dataloaders, model_name):
    losses = {lang: [] for lang in LANGUAGES}

    # Iterate over the dataset for each language and compute the cross-entropy loss per batch
    model.eval().to(device)
    for lang, dataloader in eval_dataloaders.items():
        loss = 0
        batch_step = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss += outputs.loss
                batch_step += 1

            if ((batch_step + 1) % 50) == 0:
                wandb.log({f"{lang}-{model_name}-batch-loss": outputs.loss})

        # Store the loss
        losses[lang] = loss / len(dataloader)

    # Example: Printing the average loss for each language
    for lang, loss in losses.items():
        print(f"Average loss for {lang}: {loss}")
        wandb.log({f"{lang}-{model_name}-avg-loss": loss})

########################################################
# Entry point
########################################################
if __name__ == "__main__":
    run_task3()
