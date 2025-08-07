import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

def calculate_dice(pred, target, smooth=1e-6):
    """
    计算Dice系数
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        dice: Dice系数
    """
    # 确保输入是二值化的
    pred = pred.astype(np.bool)
    target = target.astype(np.bool)
    
    # 计算交集和并集
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

def calculate_iou(pred, target, smooth=1e-6):
    """
    计算IoU(交并比)
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        iou: IoU值
    """
    # 确保输入是二值化的
    pred = pred.astype(np.bool)
    target = target.astype(np.bool)
    
    # 计算交集和并集
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def calculate_precision(pred, target):
    """
    计算精确率
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        
    返回:
        precision: 精确率
    """
    # 确保输入是二值化的
    pred = pred.flatten().astype(np.int32)
    target = target.flatten().astype(np.int32)
    
    # 计算精确率
    precision = precision_score(target, pred, average='binary', zero_division=1)
    
    return precision

def calculate_recall(pred, target):
    """
    计算召回率
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        
    返回:
        recall: 召回率
    """
    # 确保输入是二值化的
    pred = pred.flatten().astype(np.int32)
    target = target.flatten().astype(np.int32)
    
    # 计算召回率
    recall = recall_score(target, pred, average='binary', zero_division=1)
    
    return recall

def calculate_specificity(pred, target):
    """
    计算特异性(True Negative Rate)
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        
    返回:
        specificity: 特异性
    """
    # 确保输入是二值化的
    pred = pred.flatten().astype(np.bool)
    target = target.flatten().astype(np.bool)
    
    # 计算真阴性和假阳性
    tn = np.sum(np.logical_and(~pred, ~target))
    fp = np.sum(np.logical_and(pred, ~target))
    
    # 计算特异性
    specificity = tn / (tn + fp + 1e-6)
    
    return specificity

def calculate_hausdorff95(pred, target):
    """
    计算95%的Hausdorff距离
    
    参数:
        pred: 预测的分割掩码(二值化后)
        target: 真实分割掩码
        
    返回:
        hd95: 95%的Hausdorff距离
    """
    # 如果预测或真实掩码为空，返回0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 0.0
    
    # 确保输入是二值化的
    pred = pred.astype(np.bool)
    target = target.astype(np.bool)
    
    # 提取轮廓(边界像素)
    pred_border = np.logical_xor(pred, np.logical_and(
        pred, 
        np.logical_and(
            np.logical_and(
                np.roll(pred, 1, axis=0),
                np.roll(pred, 1, axis=1)
            ),
            np.logical_and(
                np.roll(pred, -1, axis=0),
                np.roll(pred, -1, axis=1)
            )
        )
    ))
    target_border = np.logical_xor(target, np.logical_and(
        target,
        np.logical_and(
            np.logical_and(
                np.roll(target, 1, axis=0),
                np.roll(target, 1, axis=1)
            ),
            np.logical_and(
                np.roll(target, -1, axis=0),
                np.roll(target, -1, axis=1)
            )
        )
    ))
    
    # 获取轮廓点坐标
    pred_coords = np.column_stack(np.where(pred_border))
    target_coords = np.column_stack(np.where(target_border))
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    
    # 检查维度是否匹配，如果不匹配则填充或截断
    if pred_coords.shape[1] != target_coords.shape[1]:
        # 取较小的维度作为共同维度
        min_dim = min(pred_coords.shape[1], target_coords.shape[1])
        pred_coords = pred_coords[:, :min_dim]
        target_coords = target_coords[:, :min_dim]
    
    # 计算定向Hausdorff距离
    d1 = directed_hausdorff(pred_coords, target_coords)[0]
    d2 = directed_hausdorff(target_coords, pred_coords)[0]
    
    # 取两个方向的最大值
    hd = max(d1, d2)
    
    return hd

def calculate_metrics(pred, target):
    """
    计算所有评估指标
    
    参数:
        pred: 预测的分割掩码
        target: 真实分割掩码
        
    返回:
        metrics: 包含所有指标的字典
    """
    # 打印输入形状和取值范围
    pred_shape = pred.shape
    target_shape = target.shape
    pred_unique = np.unique(pred)
    target_unique = np.unique(target)
    
    # 如果输入是多类别，转换为二值化表示(前景和背景)
    if pred.max() > 1 or target.max() > 1:
        pred_binary = (pred > 0).astype(np.int32)
        target_binary = (target > 0).astype(np.int32)
    else:
        pred_binary = pred
        target_binary = target
    
    # 检查二值化后的结果
    pred_binary_unique = np.unique(pred_binary)
    target_binary_unique = np.unique(target_binary)
    
    # 计算各项指标
    dice = calculate_dice(pred_binary, target_binary)
    iou = calculate_iou(pred_binary, target_binary)
    precision = calculate_precision(pred_binary, target_binary)
    recall = calculate_recall(pred_binary, target_binary)
    specificity = calculate_specificity(pred_binary, target_binary)
    hd95 = calculate_hausdorff95(pred_binary, target_binary)
    
    # 返回指标字典
    metrics = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'hd95': hd95,
        'pred_shape': pred_shape,
        'target_shape': target_shape,
        'pred_unique': pred_unique,
        'target_unique': target_unique,
        'pred_binary_unique': pred_binary_unique,
        'target_binary_unique': target_binary_unique
    }
    
    return metrics 