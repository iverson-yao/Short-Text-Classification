import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd

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
    'train': './tagmynews.tsv'
}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['text', 'extra_info', 'label'])

# 将Dataset转换为Pandas DataFrame
df = dataset['train'].to_pandas()

# 删除第一列为空的行
df = df[df['text'].notna() & (df['text'].str.strip() != '')]

# 创建标签映射
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_mapping)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 将Pandas DataFrame转换回datasets对象
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 加载本地模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=len(label_mapping))
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=30,
        return_tensors="pt"
    )

# 应用预处理
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# 设置格式为PyTorch张量
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.tensor([int(f['label']) for f in data])  # 确保标签是整数张量
    },
    compute_metrics=lambda pred: {
        'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
        'f1': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[2],
        'precision': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[0],
        'recall': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[1]
    }
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model(finetuned_model_path)

print(f"Model saved to {finetuned_model_path}")

# 在完整测试集上进行评估
eval_results = trainer.evaluate(tokenized_test_dataset)

print(f"Evaluation results on the test set: {eval_results}")