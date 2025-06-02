# PURPOSE: 
# Microbiome data analysis using LLM models
#   1. input is a csv file, column "text" is sentences for microbiome data in each individual
#   2. output is the last hidden layer from LLM.
#
# Example:
# $ python microbiome_zeroshot.py -m /data/qijun/meta-llama/Llama-3.1-8B -i mars_prompt_taxa.csv -o mars_llama3_zeroshot_taxa
# $ python microbiome_zeroshot.py -m /data/qijun/gemma-7b -i mars_prompt_taxa.csv -o mars_gemma_zeroshot_taxa



import os
import datetime
import argparse
import math
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, get_peft_model

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def main(args):

    model_name = args.model_path
    output_csv = args.output_csv
    input_csv = args.input_csv


    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    ## Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ## Use downloaded checkpoint as model
    print(f"Using model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    ## Create model
    print("Creating model...")
    model = AutoModel.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=True
    )

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0    # disable dropout

    ## Create tokenizer
    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token='<pad>'
    )

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    ## Load dataset
    print(f"Loading dataset...")
    data_df = pd.read_csv(input_csv)
    data_df.columns = ["sample_id", "text"]


    ## Tokenize the text
    print("Tokenizing the text...")
    def tokenize_text(text):
        return tokenizer(text, truncation=True, padding="max_length", max_length=4096, return_tensors="pt")
    tokenized_data = []
    for idx, row in data_df.iterrows():
        tokenized_data.append(tokenize_text(row.text))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r = 8,
        lora_alpha = 8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias = "none",
        lora_dropout=0.1,  
        task_type = "SEQ_CLS"
    ) 

    model = get_peft_model(model, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.hf_device_map[""] = model.device

    print("Generating outputs...")
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs4 = []
    outputs5 = []

    for i in range(len(tokenized_data)):
        input_ids = tokenized_data[i]['input_ids'].to('cuda')
        attention_mask = tokenized_data[i]['attention_mask'].to('cuda')
        with torch.no_grad():
            output = model(
                input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True,
                use_cache=True
                )
        outputs1.append(output.hidden_states[-1][0,:,:].cpu().numpy().mean(axis=1))
        outputs2.append(output.hidden_states[-2][0,:,:].cpu().numpy().mean(axis=1))
        outputs3.append(output.hidden_states[-3][0,:,:].cpu().numpy().mean(axis=1))
        outputs4.append(output.hidden_states[-4][0,:,:].cpu().numpy().mean(axis=1))
        outputs5.append(output.hidden_states[-5][0,:,:].cpu().numpy().mean(axis=1))
        if i % 10 == 0:
            print(f"Processed {i} reports")

    pd.DataFrame(np.array(outputs1)).to_csv(f'{output_csv}_last1.csv')
    pd.DataFrame(np.array(outputs2)).to_csv(f'{output_csv}_last2.csv')
    pd.DataFrame(np.array(outputs3)).to_csv(f'{output_csv}_last3.csv')
    pd.DataFrame(np.array(outputs4)).to_csv(f'{output_csv}_last4.csv')
    pd.DataFrame(np.array(outputs5)).to_csv(f'{output_csv}_last5.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-m', '--model_path',
                        help='the path for model checkpoint used in this script',
                        type=str,
                        default='')
    parser.add_argument('-i', '--input_csv',
                        help='the path for input file in this script',
                        type=str,
                        default='')
    parser.add_argument('-o', '--output_csv',
                        help='the path for output file in this script',
                        type=str,
                        default='')

    args = parser.parse_args()

    main(args)
