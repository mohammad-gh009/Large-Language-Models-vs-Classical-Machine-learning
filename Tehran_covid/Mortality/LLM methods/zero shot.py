import pandas as pd
import time
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import notebook_login

notebook_login()


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


dataset = load_dataset("Moreza009/internal_LLM_num")


train_ds = dataset["test"]


t = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")


start_time = time.time()

list_results = []
count = 0

for i in train_ds["patient medical hidtory"]:
    sequence_to_classify = t(i)
    candidate_labels = [1, 0]
    classing = classifier(sequence_to_classify, candidate_labels)
    scores = classing["scores"]
    highest_value_position = scores.index(max(scores))

    result = classing["labels"][highest_value_position]
    list_results.append(result)
    count += 1
    print(count)

print(list_results)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")


df = pd.DataFrame(
    {"y_pred": list_results, "y_true": train_ds["Inhospital Mortality"]})


df.to_excel("zero_shot_internal.xlsx", index=False)


dataset = load_dataset("Moreza009/External_validation_num")
train_ds = dataset["train"]


start_time = time.time()

list_results = []
count = 0

for i in train_ds["patient medical hidtory"]:
    sequence_to_classify = t(i)
    candidate_labels = [1, 0]
    classing = classifier(sequence_to_classify, candidate_labels)
    scores = classing["scores"]
    highest_value_position = scores.index(max(scores))

    result = classing["labels"][highest_value_position]
    list_results.append(result)
    count += 1
    print(count)

print(list_results)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")


df = pd.DataFrame(
    {"y_pred": list_results, "y_true": train_ds["Inhospital Mortality"]})


df.to_excel("zero_shot_external.xlsx", index=False)
