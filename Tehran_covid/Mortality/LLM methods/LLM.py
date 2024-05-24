import pandas as pd
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "HuggingFaceH4/zephyr-7b-beta"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


data = load_dataset("Moreza009/Internal_validation")
data = data.map(
    lambda samples: tokenizer(samples["patient medical hidtory"]), batched=True
)


def merge_columns(example):
    example["prediction"] = (
        "does the patient survive or die based on the provided medical history? patient history is : "
        + example["patient medical hidtory"]
        + " ----->: "
        + str(example["Inhospital Mortality"])
    )
    return example


data["train"] = data["train"].map(merge_columns)
data["train"]["prediction"][:5]


# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)
# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()


def merge_columns(example):
    example["prediction"] = (
        "does the patient survive or die based on the provided medical history? patient history is : "
        + example["patient medical hidtory"]
        + " ----->: "
        + str(example["Inhospital Mortality"])
    )
    return example


data["test"] = data["test"].map(merge_columns)
data["test"]["prediction"][:5]


y_pred = []
for i in data["test"]["prediction"]:
    device = "cuda:0"

    inputs = tokenizer(i, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1)
    if "survives" in (tokenizer.decode(outputs[0], skip_special_tokens=True)):
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_pred)


y_true = []
for i in data["test"]["prediction"]:
    if "survives" in i:
        y_true.append(1)
    else:
        y_true.append(0)


result = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})


result.to_excel("zephyr_Internal_fine_tuning_results.xlsx", index=False)
