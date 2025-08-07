import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import random
from scipy.ndimage import rotate, shift
from skimage.transform import resize
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainTumorDataset(Dataset):
    """脑肿瘤分割数据集"""
    
    def __init__(self, data_list, transform=None, crop_size=None, center_crop=False, 
                 augment=False, mode='train', normalize=True):
        """
        初始化脑肿瘤数据集
        
        参数:
            data_list (list): 包含图像和掩码路径的列表
            transform (callable, optional): 转换操作
            crop_size (tuple, optional): 裁剪大小 (高度, 宽度)
            center_crop (bool): 是否以肿瘤为中心裁剪
            augment (bool): 是否使用数据增强
            mode (str): 'train', 'val' 或 'test'
            normalize (bool): 是否进行归一化
        """
        self.data_list = data_list
        self.transform = transform
        self.crop_size = crop_size
        self.center_crop = center_crop
        self.augment = augment
        self.mode = mode
        self.normalize = normalize
        
        # 读取数据列表
        self.samples = []
        with open(data_list, 'r') as f:
            for line in f:
                img_path, mask_path = line.strip().split(',')
                if mask_path:  # 只有在有掩码的情况下才添加
                    self.samples.append((img_path, mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path):
        """加载图像，支持多种格式"""
        if not path:  # 如果路径为空
            return None
            
        try:
            if path.endswith('.npy'):
                # 如果是numpy文件，直接加载
                image = np.load(path)
            elif path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # 使用OpenCV加载常见图像格式
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
                # 检查图像是否加载成功
                if image is None:
                    print(f"警告: 无法加载图像 {path}")
                    return np.zeros((256, 256), dtype=np.uint8)
            else:
                # 未知格式
                print(f"警告: 不支持的图像格式 {path}")
                return np.zeros((256, 256), dtype=np.uint8)
                
            return image
            
        except Exception as e:
            print(f"加载图像 {path} 时出错: {str(e)}")
            # 返回空图像而不是中断程序
            return np.zeros((256, 256), dtype=np.uint8)
    
    def _preprocess(self, image, mask=None):
        """预处理图像和掩码"""
        # 确保输入格式正确
        if image.ndim == 2:
            image = image[..., np.newaxis]
        
        # 归一化
        if self.normalize:
            if image.dtype == np.uint16:
                # 对16位图像归一化到0-1
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                # 对8位图像归一化到0-1
                image = image.astype(np.float32) / 255.0
            else:
                # 已经是浮点型，确保范围在0-1
                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        # 处理掩码 (如果有)
        if mask is not None:
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # 确保掩码是二值的
            mask = (mask > 0).astype(np.uint8)
            
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
        
        return image, mask
    
    def _tumor_center_crop(self, image, mask, crop_size):
        """以肿瘤为中心进行裁剪"""
        if mask is None or np.max(mask) == 0:
            # 如果没有肿瘤掩码或掩码为空，使用中心裁剪
            h, w = image.shape[:2]
            crop_h, crop_w = crop_size
            start_h = max(0, h // 2 - crop_h // 2)
            start_w = max(0, w // 2 - crop_w // 2)
        else:
            # 找到肿瘤的中心
            indices = np.where(mask > 0)
            if len(indices[0]) == 0:
                # 如果掩码为空，使用中心裁剪
                h, w = image.shape[:2]
                crop_h, crop_w = crop_size
                start_h = max(0, h // 2 - crop_h // 2)
                start_w = max(0, w // 2 - crop_w // 2)
            else:
                # 计算肿瘤中心
                center_h = int(np.mean(indices[0]))
                center_w = int(np.mean(indices[1]))
                
                # 计算裁剪区域
                crop_h, crop_w = crop_size
                start_h = max(0, center_h - crop_h // 2)
                start_w = max(0, center_w - crop_w // 2)
        
        # 确保裁剪区域不超出图像边界
        h, w = image.shape[:2]
        end_h = min(h, start_h + crop_h)
        end_w = min(w, start_w + crop_w)
        
        # 裁剪图像和掩码
        cropped_image = image[start_h:end_h, start_w:end_w].copy()
        
        # 如果裁剪后的大小不等于指定的大小，使用调整大小
        if cropped_image.shape[0] != crop_h or cropped_image.shape[1] != crop_w:
            cropped_image = cv2.resize(cropped_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        if mask is not None:
            cropped_mask = mask[start_h:end_h, start_w:end_w].copy()
            if cropped_mask.shape[0] != crop_h or cropped_mask.shape[1] != crop_w:
                cropped_mask = cv2.resize(cropped_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        else:
            cropped_mask = None
        
        return cropped_image, cropped_mask
    
    def _apply_augmentation(self, image, mask):
        """应用数据增强"""
        # 使用albumentations进行增强
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            else:
                augmented = self.transform(image=image)
                return augmented['image'], None
                
        # 如果没有指定transform，则使用默认增强
        elif self.augment and self.mode == 'train':
            # 创建一个基本的增强管道
            aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
                A.ElasticTransform(alpha=50, sigma=10, p=0.5),
                A.GridDistortion(p=0.5),
            ])
            
            if mask is not None:
                augmented = aug(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            else:
                augmented = aug(image=image)
                return augmented['image'], None
        
        # 不进行增强
        return image, mask
    
    def __getitem__(self, idx):
        """获取数据集的一个样本"""
        img_path, mask_path = self.samples[idx]
        
        # 将Windows风格的路径分隔符替换为Linux风格
        img_path = img_path.replace('\\', '/')
        if mask_path:
            mask_path = mask_path.replace('\\', '/')
        
        # 加载图像和掩码
        image = self._load_image(img_path)
        mask = self._load_image(mask_path) if mask_path else None
        
        # 预处理
        image, mask = self._preprocess(image, mask)
        
        # 如果指定了裁剪大小，则进行裁剪
        if self.crop_size is not None:
            if self.center_crop:
                image, mask = self._tumor_center_crop(image, mask, self.crop_size)
            else:
                # 简单的中心裁剪
                h, w = image.shape[:2]
                crop_h, crop_w = self.crop_size
                start_h = max(0, h // 2 - crop_h // 2)
                start_w = max(0, w // 2 - crop_w // 2)
                end_h = min(h, start_h + crop_h)
                end_w = min(w, start_w + crop_w)
                
                image = image[start_h:end_h, start_w:end_w].copy()
                if mask is not None:
                    mask = mask[start_h:end_h, start_w:end_w].copy()
        
        # 应用数据增强
        image, mask = self._apply_augmentation(image, mask)
        
        # 转换为PyTorch张量
        if not isinstance(image, torch.Tensor):
            # 确保图像是3D的 (H, W, C)
            if image.ndim == 2:
                image = image[..., np.newaxis]  # 添加通道维度
            
            # 将图像从 (H, W, C) 转换为 (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                # 确保掩码是2D的，如果是3D则取第一个通道
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                # 然后添加通道维度
                mask = torch.from_numpy(mask).long().unsqueeze(0)
            else:
                # 如果已经是张量，确保有批次维度
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    # 如果通道在最后，转置
                    if mask.shape[2] == 1:
                        mask = mask.permute(2, 0, 1)
                        
            return {'image': image, 'mask': mask, 'path': img_path}
        else:
            return {'image': image, 'path': img_path}


def get_transforms(mode='train', crop_size=(256, 256)):
    """获取不同模式的变换"""
    # 如果crop_size是单个整数，则创建一个正方形的尺寸元组
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    if mode == 'train':
        return A.Compose([
            A.Resize(height=crop_size[0], width=crop_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=50, sigma=10, p=0.5),
            A.GridDistortion(p=0.5),
            ToTensorV2()
        ])
    elif mode == 'val' or mode == 'test':
        return A.Compose([
            A.Resize(height=crop_size[0], width=crop_size[1]),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"不支持的模式: {mode}")

def create_dataloader(data_list, batch_size=8, mode='train', crop_size=(256, 256), 
                     center_crop=True, num_workers=4, pin_memory=True):
    """创建数据加载器"""
    # 如果crop_size是单个整数，则创建一个正方形的尺寸元组
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
        
    # 获取适当的变换
    transform = get_transforms(mode=mode, crop_size=crop_size)
    
    # 创建数据集
    dataset = BrainTumorDataset(
        data_list=data_list,
        transform=transform,
        crop_size=crop_size,
        center_crop=center_crop,
        augment=(mode == 'train'),
        mode=mode,
        normalize=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )
    
    return dataloader 