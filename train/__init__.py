from .trainer import BrainTumorSegTrainer, train_model
from .train import main, train_fold

__all__ = [
    'BrainTumorSegTrainer',
    'train_model',
    'main',
    'train_fold'
]

# 将train目录标记为Python包
# 这样就可以使用 'from train.trainer import train_model' 这样的导入语句 