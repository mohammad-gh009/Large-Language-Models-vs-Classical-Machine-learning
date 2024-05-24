from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset


data = load_dataset("Moreza009/Internal_validation")


def merge_columns(example):
    example["prediction"] = example["patient medical hidtory"] + \
        " ----->: " + str(example["Inhospital Mortality"])
    return example


data['train'] = data['train'].map(merge_columns)


documents = data['train']["prediction"]

embeddings = HuggingFaceEmbeddings()

knowledge_base = FAISS.from_texts(documents, embeddings)


def merge_columns(example):
    example["prediction"] = example["patient medical hidtory"] + \
        " ----->: " + str(example["Inhospital Mortality"])
    return example


data['test'] = data['test'].map(merge_columns)


data['test']["prediction"]


y_pred = []
for i in data['test']["prediction"]:
    documents = knowledge_base.similarity_search(i, k=1)
    characters = [doc.page_content for doc in documents]
    for doc, char in zip(documents, characters):
        string = f"{char}"
        if "survives" in string:
            y_pred.append(1)
        else:
            y_pred.append(0)
print(y_pred)


y_true = []
for i in data['test']["prediction"]:
    if "survives" in i:
        y_true.append(1)
    else:
        y_true.append(0)
result = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})


result.to_excel("Vector_database_internal_results.xlsx", index=False)
