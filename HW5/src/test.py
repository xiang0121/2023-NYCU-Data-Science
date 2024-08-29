from datasets import load_dataset
from transformers import AutoTokenizer  
import json


train_dataset = load_dataset("json", data_files = "./train.json")["train"]
train_dataset = train_dataset.filter(lambda x: x["body"] != None)
train_dataset = train_dataset.train_test_split(test_size=0.1, seed=87)["train"]
val_dataset = train_dataset.train_test_split(test_size=0.1, seed=87)["test"]

test_dataset = load_dataset("json", data_files = "./test.json")["train"]

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Preprocess
prefix = "summarize: "
def preprocess_function(examples):
    inputs = [prefix + doc if doc != None else '' for doc in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    labels = tokenizer(text_target=examples["title"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def test_preprocess_function(examples):
    inputs = [prefix + doc if doc != None else '' for doc in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    return model_inputs

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=checkpoint)

import evaluate
rouge = evaluate.load("rouge", rouge_type=["rouge1", "rouge2", "rougeL"])
bertscore = evaluate.load("bertscore")
# Evalute
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_result["gen_len"] = np.mean(prediction_lens)
    result = {f"rouge_{key}": value for key, value in rouge_result.items()}
    result["precision"] = np.mean(bertscore_result["precision"])
    result["recall"] = np.mean(bertscore_result["recall"])
    result["f1"] = np.mean(bertscore_result["f1"])

    return result

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(test_preprocess_function, batched=True)

# Train
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
lr = 2e-5

training_args = Seq2SeqTrainingArguments(
    output_dir="./my_model",
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=True)

pred, _, _ = trainer.predict(tokenized_test_dataset)
decoded_preds = tokenizer.batch_decode(pred, skip_special_tokens=True)

output = []

for i in range(len(decoded_preds)):
    output.append({"title": decoded_preds[i]})


with open("submission.json", "w") as output_file:
    for dict in output:
        output_file.write(json.dumps(dict) + '\n')
