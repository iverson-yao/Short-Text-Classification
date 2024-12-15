# python学习日记
import torch
print(torch.cuda.is_available())  # 如果返回 True，说明 CUDA 可用
print(torch.cuda.current_device())  # 显示当前 GPU 的设备编号
print(torch.cuda.device_count())  # 显示可用的 GPU 数量
print(torch.cuda.get_device_name(0))  # 显示第一个 GPU 的名称
import torchtext
print(torchtext.__version__)