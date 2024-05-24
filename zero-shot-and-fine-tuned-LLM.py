# install the followinf libraries -q -U accelerate=='0.25.0' peft=='0.7.1' bitsandbytes=='0.41.3.post2' transformers=='4.36.1' trl=='0.7.4'


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# silence warnings
import warnings
warnings.filterwarnings("ignore")


# import libraries

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# load training and test sets

train = "X_train_u_F.xlsx"
test = "X_test_u_F.xlsx"


train = pd.read_excel(train)
test = pd.read_excel(test)

X = train["patient medical hidtory"]
y = train["Inhospital Mortality"]


# seprate 0.2 of training set for evaluation
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42
)

eval = pd.concat([X_eval, y_eval], axis=1)


# preprocess and convert data into text


def generate_prompt(data_point):
    return f"""
            [INST]You're tasked with analyzing the present symptoms, past medical history, 
            laboratory data, age, and gender of COVID patients to determine their outcome, 
            which is enclosed in square brackets. Your goal is to predict whether the patient will "survive" or "die" based on this information.[/INST]

            [{data_point["patient medical hidtory"]}] = {data_point["Inhospital Mortality"]}
            """.strip()


def generate_test_prompt(data_point):
    return f"""
            [INST]You're tasked with analyzing the present symptoms, past medical history, 
            laboratory data, age, and gender of COVID patients to determine their outcome, 
            which is enclosed in square brackets. Your goal is to predict whether the patient will "survive" or "die" based on this information.[/INST]

            [{data_point["patient medical hidtory"]}] = """.strip()


X_train = pd.DataFrame(
    train.apply(generate_prompt, axis=1), columns=["patient medical hidtory"]
)
X_eval = pd.DataFrame(
    eval.apply(generate_prompt, axis=1), columns=["patient medical hidtory"]
)

y_true = test["Inhospital Mortality"]
X_test = pd.DataFrame(
    test.apply(generate_test_prompt, axis=1), columns=["patient medical hidtory"]
)

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)


# load model , tokenizer and set BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left",
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


# define a function to make a list of y predictions


def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["patient medical hidtory"]
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2,
            temperature=0.0,
        )
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]["generated_text"].split("=")[-1].lower()
        if "die" in answer:
            y_pred.append("die")
        else:
            y_pred.append("survive")
    return y_pred


# mortality is predicted  by zero shot classification method

y_pred = predict(X_test, model, tokenizer)


# save predictions to csv file

evaluation = pd.DataFrame(
    {"text": X_test["patient medical hidtory"], "y_true": y_true, "y_pred": y_pred},
)
evaluation.to_csv("zero_shot_predictions.csv", index=False)


# set peft config and trainig arguments

model.config.max_new_tokens = 2
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="patient medical hidtory",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=2500,
)


# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained("trained-model")


# use fine-tuned model to predict test set

y_pred = predict(X_test, model, tokenizer)


# save predictions to csv file

evaluation = pd.DataFrame(
    {"text": X_test["text"], "y_true": y_true, "y_pred": y_pred},
)
evaluation.to_csv("test_predictions.csv", index=False)
