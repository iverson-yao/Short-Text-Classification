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