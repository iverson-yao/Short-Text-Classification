import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")

# 指定本地模型路径
local_model_path = "./offline_model"
finetuned_model_path = "./offline_model_finetuned_small"

# 加载数据集
data_files = {
    'train': 'C:/Users/iverson/Downloads/train-00000-of-00001.parquet',
    'test': 'C:/Users/iverson/Downloads/test-00000-of-00001.parquet'
}
dataset = load_dataset('parquet', data_files=data_files)

# 加载本地模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['text'],  # 确认字段名为'text'
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"  # 返回PyTorch张量
    )

# 应用预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 移除不必要的列（如原始文本）
columns_to_remove = ['text']
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

# 设置格式为PyTorch张量
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 创建小规模训练和验证数据集
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # 使用新参数名
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    no_cuda=False,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,  # 使用小规模训练数据集
    eval_dataset=small_eval_dataset,    # 使用小规模评估数据集
    tokenizer=tokenizer,
    compute_metrics=lambda pred: {
        'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
        'f1': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[2],
        'precision': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[0],
        'recall': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[1]
    }
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model(finetuned_model_path)

print(f"Model saved to {finetuned_model_path}")

# 在完整测试集上进行评估

# 重新加载保存的模型
model_for_evaluation = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path)
tokenizer_for_evaluation = AutoTokenizer.from_pretrained(finetuned_model_path)

# 初始化用于评估的Trainer，设置eval_strategy为'no'，因为我们在evaluate方法中明确指定评估数据集
eval_args = TrainingArguments(
    output_dir='./results',
    do_eval=False,  # 不执行自动评估
    per_device_eval_batch_size=8,
)

eval_trainer = Trainer(
    model=model_for_evaluation,
    args=eval_args,
    tokenizer=tokenizer_for_evaluation,
    compute_metrics=lambda pred: {
        'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
        'f1': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[2],
        'precision': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[0],
        'recall': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='binary')[1]
    }
)

# 使用完整的测试集进行评估
full_test_dataset = tokenized_datasets["test"]

eval_results = eval_trainer.evaluate(full_test_dataset)

print(f"Evaluation results on the full test set: {eval_results}")