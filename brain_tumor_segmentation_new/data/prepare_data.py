import os
import argparse
import numpy as np
import h5py
import scipy.io as sio
from tqdm import tqdm
import cv2
from pathlib import Path
from sklearn.model_selection import KFold

def normalize_image(image):
    """Z-score标准化，排除背景区域"""
    mask = image > 10  # 定义一个简单的背景阈值
    if np.sum(mask) > 0:
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[mask] = (image[mask] - mean) / (std + 1e-8)
        return normalized
    else:
        return (image - np.mean(image)) / (np.std(image) + 1e-8)

def mat_to_npy(mat_path, output_dir, keep_16bit=True):
    """将.mat文件转换为.npy格式"""
    try:
        # 尝试使用scipy.io加载
        mat_data = sio.loadmat(mat_path)
        cjdata = mat_data['cjdata']
    except NotImplementedError:
        # 使用h5py加载MATLAB v7.3文件
        with h5py.File(mat_path, 'r') as f:
            # 对于h5py读取的文件，结构可能不同
            cjdata = {}
            # 读取标签
            if 'cjdata/label' in f:
                cjdata['label'] = f['cjdata/label'][()]
            # 读取PID
            if 'cjdata/PID' in f:
                # 对于h5py，字符串需要特殊处理
                pid_data = f['cjdata/PID'][()]
                if isinstance(pid_data, np.ndarray):
                    if pid_data.dtype.kind == 'S':  # 如果是字节字符串
                        cjdata['PID'] = pid_data.astype(str)
                    else:
                        # 处理数字数组
                        cjdata['PID'] = str(pid_data)
                else:
                    cjdata['PID'] = str(pid_data)
            # 读取图像
            if 'cjdata/image' in f:
                cjdata['image'] = f['cjdata/image'][()]
            # 读取肿瘤掩码
            if 'cjdata/tumorMask' in f:
                cjdata['tumorMask'] = f['cjdata/tumorMask'][()]
        
        # 读取字段
        image = cjdata['image']
        label = cjdata['label']
        pid = cjdata['PID']
        tumor_mask = cjdata['tumorMask'] if 'tumorMask' in cjdata else None
        
    # 创建输出目录结构
    class_dir = os.path.join(output_dir, f"class_{label}")
    os.makedirs(class_dir, exist_ok=True)
    
    # 提取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(mat_path))[0]
    
    # 如果保持16位精度，则不进行归一化和转换
    if keep_16bit:
        # 确保图像为16位格式
        if image.dtype != np.uint16:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 65535).astype(np.uint16)
    else:
        # 否则进行归一化并转换为8位
        image = normalize_image(image)
        image = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)
    
    # 保存图像和肿瘤掩码
    np.save(os.path.join(class_dir, f"{file_name}_image.npy"), image)
    if tumor_mask is not None:
        np.save(os.path.join(class_dir, f"{file_name}_mask.npy"), tumor_mask.astype(np.uint8))
    
    # 返回元数据字典
    return {
        "file_name": file_name,
        "label": int(label.item()) if isinstance(label, np.ndarray) and label.size == 1 else label,
        "pid": pid,
        "shape": image.shape,
        "path": os.path.join(class_dir, f"{file_name}_image.npy"),
        "mask_path": os.path.join(class_dir, f"{file_name}_mask.npy") if tumor_mask is not None else None
    }

def convert_to_images(mat_path, output_dir, format='jpg'):
    """将.mat文件转换为图像格式（jpg/png等）"""
    try:
        # 尝试使用scipy.io加载
        mat_data = sio.loadmat(mat_path)
        cjdata = mat_data['cjdata']
        
        # 读取字段
        image = cjdata['image'][0, 0]
        label = cjdata['label'][0, 0][0, 0]
        tumor_mask = cjdata['tumorMask'][0, 0] if 'tumorMask' in cjdata.dtype.names else None
        
    except:
        # 如果失败，尝试使用h5py
        f = h5py.File(mat_path, 'r')
        cjdata = f['cjdata']
        
        # 读取字段
        image = np.array(cjdata['image']).T
        label = int(np.array(cjdata['label'])[0, 0])
        tumor_mask = np.array(cjdata['tumorMask']).T if 'tumorMask' in cjdata else None
    
    # 创建输出目录结构
    class_dir = os.path.join(output_dir, f"class_{label}")
    os.makedirs(class_dir, exist_ok=True)
    
    # 提取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(mat_path))[0]
    
    # 归一化图像到0-255范围
    im1 = image.astype(np.float64)
    min1 = np.min(im1)
    max1 = np.max(im1)
    im = ((im1 - min1) / (max1 - min1 + 1e-8) * 255).astype(np.uint8)
    
    # 保存图像
    image_path = os.path.join(class_dir, f"{file_name}.{format}")
    cv2.imwrite(image_path, im)
    
    # 如果有肿瘤掩码，也保存掩码
    if tumor_mask is not None:
        mask_path = os.path.join(class_dir, f"{file_name}_mask.{format}")
        cv2.imwrite(mask_path, (tumor_mask * 255).astype(np.uint8))
    
    return {
        "file_name": file_name,
        "label": label,
        "path": image_path,
        "mask_path": mask_path if tumor_mask is not None else None
    }

def create_dataset_split(metadata, output_dir, n_splits=5, test_size=0.1, val_size=0.1, random_state=42):
    """创建训练、验证和测试集划分，基于患者ID"""
    # 按患者ID分组
    patients = {}
    for item in metadata:
        pid = item['pid']
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(item)
    
    # 获取所有患者ID
    pids = list(patients.keys())
    
    # 创建KFold对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 生成每折的训练和测试索引
    fold_data = []
    for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(pids)):
        # 将训练集进一步分为训练和验证
        train_val_pids = [pids[i] for i in train_idxs]
        test_pids = [pids[i] for i in test_idxs]
        
        # 计算验证集大小
        n_val = int(len(train_val_pids) * (val_size / (1 - test_size)))
        
        # 划分训练和验证集
        val_pids = train_val_pids[:n_val]
        train_pids = train_val_pids[n_val:]
        
        # 收集每个集合的样本
        train_items = [item for pid in train_pids for item in patients[pid]]
        val_items = [item for pid in val_pids for item in patients[pid]]
        test_items = [item for pid in test_pids for item in patients[pid]]
        
        fold_data.append({
            'fold': fold_idx + 1,
            'train': train_items,
            'val': val_items,
            'test': test_items
        })
    
    # 保存划分信息
    np.save(os.path.join(output_dir, 'dataset_split.npy'), fold_data)
    
    # 为每个折创建索引文件
    for fold in fold_data:
        fold_dir = os.path.join(output_dir, f"fold_{fold['fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            with open(os.path.join(fold_dir, f"{split}.txt"), 'w') as f:
                for item in fold[split]:
                    f.write(f"{item['path']},{item.get('mask_path', '')}\n")
    
    return fold_data

def process_dataset(input_dir, output_dir, img_format='npy', n_splits=5):
    """处理整个数据集，转换格式并创建划分"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有.mat文件
    mat_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mat') and 'cvind' not in file:
                mat_files.append(os.path.join(root, file))
    
    # 转换文件并收集元数据
    metadata = []
    for mat_file in tqdm(mat_files, desc="Converting files"):
        if img_format == 'npy':
            meta = mat_to_npy(mat_file, output_dir)
        else:
            meta = convert_to_images(mat_file, output_dir, format=img_format)
        metadata.append(meta)
    
    # 创建数据集划分
    splits = create_dataset_split(metadata, output_dir, n_splits=n_splits)
    
    # 保存元数据
    with open(os.path.join(output_dir, 'metadata.csv'), 'w') as f:
        f.write("file_name,label,pid,path,mask_path\n")
        for item in metadata:
            mask_path = item.get('mask_path', '')
            f.write(f"{item['file_name']},{item['label']},{item['pid']},{item['path']},{mask_path}\n")
    
    print(f"处理完成！共转换 {len(metadata)} 个文件")
    print(f"元数据已保存至: {os.path.join(output_dir, 'metadata.csv')}")
    print(f"数据集划分已保存至: {os.path.join(output_dir, 'dataset_split.npy')}")
    
    # 统计每个类别的样本数量
    class_counts = {}
    for item in metadata:
        label = item['label']
        if isinstance(label, np.ndarray):
            label_value = int(label.item()) if label.size == 1 else str(label)
        else:
            label_value = int(label) if isinstance(label, (int, float)) else str(label)
        class_counts[label_value] = class_counts.get(label_value, 0) + 1
    
    print("\n类别统计:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="脑肿瘤数据集处理工具")
    parser.add_argument('--input_dir', type=str, required=True, help="包含.mat文件的输入目录")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录")
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'jpg', 'png'], 
                        help="输出格式，默认为npy")
    parser.add_argument('--n_splits', type=int, default=5, help="K折交叉验证的折数，默认为5")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, img_format=args.format, n_splits=args.n_splits) 