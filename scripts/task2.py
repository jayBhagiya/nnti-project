import argparse
import torch 
import h5py
import random

import datasets
import numpy as np
import transformers

MODEL_NAME = "facebook/xglm-564M"
DATASET_NAME = "facebook/flores"

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tel_Telu", # Added Telu as well
    "tam_Taml",
    "quy_Latn"
]

########################################################
# Entry point
########################################################

if __name__ == "__main__":
    # TODO: your code goes here

    # Set default device
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available and enabled!")
        default_device = torch.device("cuda") # Set the device to GPU 1
    else:
        print("CUDA (GPU support) is not available. Using CPU.")
        default_device = torch.device("cpu")

    # Loading Datasets
    dataset = {}
    for lang in LANGUAGES:
        dataset[lang] = datasets.load_dataset(DATASET_NAME, lang, trust_remote_code=True)

    # Sample 200 sentences from each languages
    sampled_dataset = {}
    for lang in LANGUAGES:
        num_samples = len(dataset[lang]['dev'])
        # Sample 200 random indices from the dataset
        random_indices = random.sample(range(num_samples), 200)
        # Extract the 200 random samples
        samples = [dataset[lang]['dev'][i] for i in random_indices]
        sampled_dataset[lang] = samples
    
    # Tokenize sentence
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'

    # Dataloader for each dataset
    BATCH_SIZE = 4 
    dataloaders = {
        lang: torch.utils.data.DataLoader(sampled_dataset[lang], batch_size=BATCH_SIZE, shuffle=True) for lang in LANGUAGES
    }

    # Load Model
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval().to(default_device)

    # Compute Embeddings
    token_embeddings = [] # To store all the embeddings for each tokens
    sentence_embeddings = []
    tokens = []
    sequences = []

    with torch.no_grad():
        for lang in LANGUAGES:
            # Get dataloader for each langugae
            print("Language:",lang)
            current_loader = dataloaders[lang]
           
            lang_token_embeddings = torch.zeros((0,25,1024)).to(default_device)
            lang_sentence_embeddings = torch.zeros((0,25,1024)).to(default_device)
            lang_tokens = []
            lang_sequences = []
            for i in range(len(current_loader)):
                sentences = next(iter(current_loader))["sentence"]
                # Save sentence sequences
                lang_sequences.extend(sentences)

                # encode the sentence 
                encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

                # get the input ids
                input_ids = encoded_sentences["input_ids"].to(default_device)

                # attention mask either holds a value of 0 or 1 depending on
                # if the token is a padded one or not.
                attention_mask = encoded_sentences["attention_mask"]

                encoded_non_pad_tokens = input_ids.reshape(-1)[attention_mask.reshape(-1).bool()]
                decoded_tokens = [tokenizer.decode([token]) for token in encoded_non_pad_tokens]
                lang_tokens.extend(decoded_tokens)

                # we need to put a label of -100 for the padded parts
                labels = input_ids.clone().to(default_device)
                labels[attention_mask == 0] = -100

                # Pass the input ids and labels to the model
                generated_output = model(input_ids=input_ids, attention_mask = attention_mask, labels=labels, output_hidden_states= True)

                # Get the embedding
                hidden_states = generated_output.hidden_states      # Tuple of length 25. Shape [batch_size, num_token, 1024] for each of 25 layers
                concatenated_hidden_states = torch.stack(hidden_states, dim =2)  # Stack the elements of tuple. Shape [batch_size, num_token, 25, 1024] 
                
                flattened_hidden_states = concatenated_hidden_states[attention_mask.bool()] # Removing the padding tokens. Shape [total_pad_removed_tokens, 25, 1024]
                lang_token_embeddings = torch.concat((lang_token_embeddings,flattened_hidden_states), dim =0) 

                # Compute the average for each sentence
                splits = attention_mask.sum(dim=1)
                sentence_emb = []
                start_idx = 0
                for split in splits:
                    end_idx = start_idx + split
                    sentence_hidden_states = flattened_hidden_states[start_idx:end_idx] # Shape [num_pad_removed_token, 25, 1024]
                    averaged_sentence_hidden_states = torch.mean(sentence_hidden_states, dim=0)  # Shape: [25, 1024]
                    sentence_emb.append(averaged_sentence_hidden_states)
                    start_idx = end_idx

                sentence_emb = torch.stack(sentence_emb, dim=0) # Shape [batch_size, 25, 1024]
                lang_sentence_embeddings = torch.concat((lang_sentence_embeddings,sentence_emb), dim =0) 

            token_embeddings.append(lang_token_embeddings) # Append embeddings of tokens of each language. lang_embeddings shape -> [total_tokens, 25, 1024]
            sentence_embeddings.append(lang_sentence_embeddings)
            tokens.append(lang_tokens)
            sequences.append(lang_sequences)
    
    # Strings needs to be encoded before saving into HDF% files
    def encode_strings(string_list):
        return [s.encode('utf-8') for s in string_list]

    # Open an HDF5 file in write mode
    with h5py.File('embeddings.h5', 'w') as f:
        # Save token embeddings 
        for i, tensor in enumerate(token_embeddings):
            f.create_dataset(f'token_embeddings/{i}', data=tensor.numpy())

        # Save tokens. Needs to be encodes for special tokens
        for i, string_list in enumerate(tokens):
            encoded_strings = encode_strings(string_list)
            f.create_dataset(f'tokens/{i}', data=np.array(encoded_strings, dtype='S'))

        # Save sentence embeddings
        for i, tensor in enumerate(sentence_embeddings):
            f.create_dataset(f'sentence_embeddings/{i}', data=tensor.numpy())

        # Save sequences. Needs to be encodes for special tokens
        for i, string_list in enumerate(sequences):
            encoded_strings = encode_strings(string_list)
            f.create_dataset(f'sequences/{i}', data=np.array(encoded_strings, dtype='S'))

    print("Embeddings saved successfully.")





