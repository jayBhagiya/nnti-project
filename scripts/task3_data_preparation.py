import datasets
from torch.utils.data import Dataset, DataLoader

DATASET_NAME = "facebook/flores" # Just for the evalution of the languages defined in the task2

def prepare_data_fine_tune_eval(languages):
    """
    During find-tuning, we need to keep track of the model's performance
    on dataset that it is not being fine-tuned on.
    """
    dataset = []
    for language in languages:
        dataset.append(datasets.load_dataset(DATASET_NAME, language, trust_remote_code=True))

    # create a dictionary, where the keys are the language names and the values
    # are the sentences inside "dev".
    eval_dataset={}
    for i, language in enumerate(languages):
        eval_dataset[language] = dataset[i]["dev"]["sentence"]

    return eval_dataset


def prepare_data_qu(dataset_path, data_files):
    """
    This function prepares the data that will be used for fine-tuning.
    """
    # load the dataset
    dataset = datasets.load_dataset(dataset_path, data_files=data_files)

    # we need to get rid of the spanish sentences because our task is not translation but token generation.
    # Remove the "es" column from each split
    dataset = dataset.map(lambda example: {"qu": example["qu"]}, remove_columns=["es"], batched=True)

    # get the train and test separately
    train_dataset = dataset["train"]
    evaluation_dataset = dataset["validation"]

    # Convert the dataset to a list of sentences, assuming each item is a dictionary with the key "qu"
    train_dataset_list = [example["qu"] for example in train_dataset]
    evaluation_dataset_list = [example["qu"] for example in evaluation_dataset]

    return train_dataset_list, evaluation_dataset_list