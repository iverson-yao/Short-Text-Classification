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