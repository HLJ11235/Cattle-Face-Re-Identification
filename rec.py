import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, average_precision_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import psutil

# 导入您的模型和配置
from model3 import mobile_vit_x_small
import config

# 设备配置
USE_CUDA = torch.cuda.is_available() and config.is_cuda
device = torch.device(config.device if USE_CUDA else "cpu")
print(f"使用设备: {device}")


# 显存监控工具
class GPUMemoryMonitor:
    """GPU显存监控和管理工具"""

    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'

    def get_memory_info(self):
        """获取当前显存信息"""
        if not self.is_cuda:
            return {"allocated": 0, "cached": 0, "total": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3  # GB
        cached = torch.cuda.memory_reserved(self.device) / 1024 ** 3  # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3  # GB
        free = total - allocated

        return {
            "allocated": allocated,
            "cached": cached,
            "total": total,
            "free": free
        }

    def print_memory_status(self, stage=""):
        """打印显存状态"""
        if not self.is_cuda:
            return

        info = self.get_memory_info()
        print(f"[{stage}] GPU显存 - 已用: {info['allocated']:.2f}GB, "
              f"缓存: {info['cached']:.2f}GB, 总计: {info['total']:.2f}GB, "
              f"可用: {info['free']:.2f}GB")

    def force_cleanup(self):
        """强制清理显存"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    def get_optimal_batch_size(self, base_size=64):
        """根据显存情况动态调整批处理大小"""
        if not self.is_cuda:
            return base_size

        info = self.get_memory_info()

        # 根据可用显存调整批处理大小
        if info['free'] < 2.0:  # 小于2GB
            return min(base_size, 8)
        elif info['free'] < 4.0:  # 小于4GB
            return min(base_size, 16)
        elif info['free'] < 6.0:  # 小于6GB
            return min(base_size, 32)
        else:
            return base_size


class MemoryEfficientEffiLibCatReID:
    """
    显存优化的EffiLibCatReID系统

    主要优化：
    - 动态批处理大小调整
    - CPU-GPU内存管理
    - 渐进式特征累积
    - 智能显存监控
    - 自动垃圾回收
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # 核心组件
        self.feature_library = None
        self.identity_labels = None
        self.pca_transformer = None
        self.compiled_extractor = None

        # 显存管理
        self.memory_monitor = GPUMemoryMonitor(device)
        self.dtype = torch.float32

        # 移除固定内存池，改为动态分配
        self.dynamic_batch_size = True
        self.max_batch_size = 64
        self.min_batch_size = 4

        # 性能监控
        self.timer = GPUAcceleratedTimer(device)

        print("🚀 内存优化EffiLibCatReID系统初始化完成")

    def setup_optimization_environment(self):
        """设置内存友好的优化环境"""
        print("⚡ 正在设置内存优化环境...")

        self.memory_monitor.print_memory_status("初始化前")

        # 1. 轻量级预编译
        self._compile_transforms_lightweight()

        # 2. 条件性JIT编译（仅在显存充足时）
        self._conditional_compile_feature_extractor()

        # 3. 清理初始化产生的临时变量
        self.memory_monitor.force_cleanup()

        self.memory_monitor.print_memory_status("优化环境设置完成")
        print("✅ 内存优化环境设置完成")

    def _compile_transforms_lightweight(self):
        """轻量级图像变换预编译"""
        print("🔧 轻量级图像变换预编译...")

        # 快速变换管道
        self.fast_resize = lambda img: cv2.resize(img, (self.config.img_s, self.config.img_s))

        # 预计算归一化参数（保持在CPU，需要时再移到GPU）
        self.norm_mean_cpu = torch.tensor(self.config.dataset_mean, dtype=self.dtype)
        self.norm_std_cpu = torch.tensor(self.config.dataset_std, dtype=self.dtype)

        print("✅ 轻量级变换预编译完成")

    def _conditional_compile_feature_extractor(self):
        """条件性JIT编译特征提取器"""
        print("⚙️ 检查JIT编译条件...")

        memory_info = self.memory_monitor.get_memory_info()

        # 只有在显存充足时才进行JIT编译
        if memory_info['free'] > 3.0:  # 大于3GB空闲显存
            try:
                print("显存充足，进行JIT编译...")

                class DCFuseViTExtractor(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.backbone = model

                    def forward(self, x):
                        return self.backbone.extract_features(x)

                extractor = DCFuseViTExtractor(self.model).eval()

                # 小尺寸dummy input进行trace
                dummy_input = torch.randn(1, 3, self.config.img_s, self.config.img_s,
                                          dtype=self.dtype, device=self.device)

                # 移动归一化参数到GPU（临时）
                norm_mean_gpu = self.norm_mean_cpu.to(self.device).view(1, 3, 1, 1)
                norm_std_gpu = self.norm_std_cpu.to(self.device).view(1, 3, 1, 1)

                dummy_norm = (dummy_input - norm_mean_gpu) / norm_std_gpu

                self.compiled_extractor = torch.jit.trace(extractor, dummy_norm)
                self.compiled_extractor.eval()

                # 清理临时变量
                del dummy_input, dummy_norm, norm_mean_gpu, norm_std_gpu, extractor
                self.memory_monitor.force_cleanup()

                print("✅ JIT编译成功")

            except Exception as e:
                print(f"⚠️ JIT编译失败，使用原始模型: {e}")
                self.compiled_extractor = None
                self.memory_monitor.force_cleanup()
        else:
            print(f"⚠️ 显存不足({memory_info['free']:.1f}GB)，跳过JIT编译")
            self.compiled_extractor = None

    def build_efficient_feature_library(self, data_dir: str, pca_dim: int = 256,
                                        base_batch_size: int = 32, save_path: str = None):
        """
        内存优化的特征库构建

        Args:
            data_dir: 训练数据目录
            pca_dim: PCA降维维度
            base_batch_size: 基础批处理大小（会动态调整）
            save_path: 保存路径
        """
        print("🏗️ 构建内存优化EffiLibCatReID特征库...")

        if save_path is None:
            save_path = self.config.save_path

        self.memory_monitor.print_memory_status("特征库构建开始")

        # 1. 数据收集
        img_paths, identity_indices, index_to_identity = self._collect_training_data(data_dir)

        # 2. 动态批量特征提取（内存优化版本）
        raw_features = self._extract_features_memory_optimized(img_paths, base_batch_size)

        # 3. 在CPU上进行PCA压缩
        compressed_features, pca_model = self._apply_pca_compression_cpu(raw_features, pca_dim)

        # 4. 批量归一化（避免大矩阵运算）
        normalized_features = self._normalize_features_chunked(compressed_features)

        # 5. 保存特征库
        self._save_feature_library(normalized_features, identity_indices, index_to_identity,
                                   pca_model, save_path)

        # 6. 轻量级加载到内存
        self._lightweight_load_to_memory(normalized_features, identity_indices, index_to_identity, pca_model)

        self.memory_monitor.print_memory_status("特征库构建完成")
        print("✅ 内存优化EffiLibCatReID特征库构建完成")
        return True

    def _collect_training_data(self, data_dir):
        """收集训练数据路径和标签映射"""
        print("📁 收集训练数据...")

        img_paths = []
        identity_indices = []
        index_to_identity = {}
        current_index = 0

        for identity_folder in tqdm(os.listdir(data_dir), desc="扫描身份文件夹"):
            identity_path = os.path.join(data_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            index_to_identity[current_index] = identity_folder

            for image_file in os.listdir(identity_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_paths.append(os.path.join(identity_path, image_file))
                    identity_indices.append(current_index)

            current_index += 1

        print(f"收集到 {len(img_paths)} 张图像，{len(index_to_identity)} 个身份")
        return img_paths, identity_indices, index_to_identity

    def _extract_features_memory_optimized(self, img_paths, base_batch_size):
        """内存优化的批量特征提取"""
        print("🔥 内存优化批量特征提取...")

        # 动态调整批处理大小
        optimal_batch_size = self.memory_monitor.get_optimal_batch_size(base_batch_size)
        print(f"动态批处理大小: {optimal_batch_size}")

        dataset = MemoryEfficientImageDataset(img_paths, self.config)
        dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False,
                                num_workers=min(2, psutil.cpu_count() // 2), pin_memory=True)

        all_features = []  # 在CPU上累积特征
        total_time = 0
        total_images = 0
        processed_batches = 0

        # 预移动归一化参数到GPU
        norm_mean_gpu = self.norm_mean_cpu.to(self.device, dtype=self.dtype).view(1, 3, 1, 1)
        norm_std_gpu = self.norm_std_cpu.to(self.device, dtype=self.dtype).view(1, 3, 1, 1)

        for batch_images in tqdm(dataloader, desc="提取特征"):
            batch_size_actual = batch_images.size(0)
            total_images += batch_size_actual
            processed_batches += 1

            start_time = self.timer.start()

            try:
                # 移动到GPU
                batch_images = batch_images.to(self.device, dtype=self.dtype, non_blocking=True)

                # 归一化
                normalized_batch = (batch_images - norm_mean_gpu) / norm_std_gpu

                # 特征提取
                if self.compiled_extractor is not None:
                    try:
                        features = self.compiled_extractor(normalized_batch)
                    except Exception as e:
                        print(f"JIT模型执行失败，回退到原始模型: {e}")
                        features = self.model.extract_features(normalized_batch)
                else:
                    features = self.model.extract_features(normalized_batch)

                # 维度调整
                if features.dim() > 2:
                    features = features.view(batch_size_actual, -1)

                # 立即移到CPU，释放GPU内存
                features_cpu = features.detach().cpu()
                all_features.append(features_cpu)

                # 清理GPU内存
                del features, batch_images, normalized_batch

                batch_time = self.timer.end(start_time)
                total_time += batch_time

                # 每处理一定数量的批次后清理显存
                if processed_batches % 10 == 0:
                    self.memory_monitor.force_cleanup()

                    # 动态调整批处理大小
                    if processed_batches % 50 == 0:
                        new_batch_size = self.memory_monitor.get_optimal_batch_size(base_batch_size)
                        if new_batch_size != optimal_batch_size:
                            print(f"调整批处理大小: {optimal_batch_size} -> {new_batch_size}")
                            optimal_batch_size = new_batch_size

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ 显存不足，自动降低批处理大小")
                    self.memory_monitor.force_cleanup()

                    # 减少批处理大小
                    optimal_batch_size = max(self.min_batch_size, optimal_batch_size // 2)

                    # 重新创建dataloader
                    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False,
                                            num_workers=min(2, psutil.cpu_count() // 2), pin_memory=True)
                    print(f"降低批处理大小到: {optimal_batch_size}")
                    continue
                else:
                    raise e

        # 清理GPU上的归一化参数
        del norm_mean_gpu, norm_std_gpu
        self.memory_monitor.force_cleanup()

        # 在CPU上拼接所有特征
        print("拼接特征向量...")
        concatenated_features = torch.cat(all_features, dim=0)

        # 清理特征列表
        del all_features
        gc.collect()

        avg_time = total_time / total_images if total_images > 0 else 0
        print(f"特征提取完成: {total_images} 张图像")
        print(f"平均时间: {avg_time:.2f} ms/图像")
        print(f"吞吐量: {1000 / avg_time:.1f} fps")

        return concatenated_features

    def _apply_pca_compression_cpu(self, features, pca_dim):
        """在CPU上应用PCA压缩，避免显存占用"""
        print(f"🗜️ CPU PCA压缩: {features.shape[1]} -> {pca_dim}")

        features_np = features.numpy()

        if features_np.shape[1] <= pca_dim:
            print("⚠️ 原始维度已小于目标维度，跳过PCA")
            return features, None

        print("在CPU上进行PCA拟合...")
        pca = PCA(n_components=pca_dim, random_state=42)

        # 分块处理大矩阵，避免内存溢出
        chunk_size = 10000
        if features_np.shape[0] > chunk_size:
            print(f"分块处理PCA，块大小: {chunk_size}")
            # 先用部分数据拟合PCA
            pca.fit(features_np[:chunk_size])

            # 分块转换所有数据
            compressed_chunks = []
            for i in range(0, features_np.shape[0], chunk_size):
                chunk = features_np[i:i + chunk_size]
                compressed_chunk = pca.transform(chunk)
                compressed_chunks.append(compressed_chunk)

            compressed_features = np.vstack(compressed_chunks)
        else:
            compressed_features = pca.fit_transform(features_np)

        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA压缩完成，保留方差: {explained_variance:.4f}")

        return torch.tensor(compressed_features, dtype=self.dtype), pca

    def _normalize_features_chunked(self, features):
        """分块归一化特征，避免大矩阵运算"""
        print("🎯 分块L2归一化...")

        chunk_size = 5000
        normalized_chunks = []

        for i in range(0, features.size(0), chunk_size):
            chunk = features[i:i + chunk_size]

            # 移到GPU进行归一化
            chunk_gpu = chunk.to(self.device, dtype=self.dtype)
            normalized_chunk = F.normalize(chunk_gpu, p=2, dim=1)

            # 移回CPU
            normalized_chunks.append(normalized_chunk.cpu())

            # 清理GPU内存
            del chunk_gpu, normalized_chunk

        self.memory_monitor.force_cleanup()

        # 拼接归一化后的特征
        normalized_features = torch.cat(normalized_chunks, dim=0)

        # 验证归一化结果
        sample_norms = torch.norm(normalized_features[:100], p=2, dim=1)
        avg_norm = sample_norms.mean().item()
        print(f"归一化验证 - 平均L2范数: {avg_norm:.6f} (应接近1.0)")

        return normalized_features

    def _save_feature_library(self, features, labels, mapping, pca_model, save_path):
        """保存特征库到磁盘"""
        print("💾 保存特征库...")

        os.makedirs(save_path, exist_ok=True)

        library_path = os.path.join(save_path, "memory_optimized_library.pkl")
        mapping_path = os.path.join(save_path, "identity_mapping.npy")
        pca_path = os.path.join(save_path, "pca_compressor.pkl")

        # 保存特征库
        with open(library_path, "wb") as f:
            pickle.dump((features, labels), f)

        # 保存身份映射
        np.save(mapping_path, mapping)

        # 保存PCA模型
        if pca_model is not None:
            with open(pca_path, "wb") as f:
                pickle.dump(pca_model, f)

        print(f"✅ 特征库已保存到: {save_path}")

    def _lightweight_load_to_memory(self, features, labels, mapping, pca_model):
        """轻量级加载到内存"""
        print("📦 轻量级加载到GPU内存...")

        # 检查显存是否足够
        feature_size_gb = features.numel() * 4 / 1024 ** 3  # float32
        memory_info = self.memory_monitor.get_memory_info()

        if memory_info['free'] > feature_size_gb + 1.0:  # 保留1GB缓冲
            self.feature_library = features.to(self.device, dtype=self.dtype)
            print(f"✅ 特征库已加载到GPU ({feature_size_gb:.2f}GB)")
        else:
            print(f"⚠️ 显存不足，特征库保持在CPU ({feature_size_gb:.2f}GB)")
            self.feature_library = features

        self.identity_labels = labels
        self.identity_mapping = mapping
        self.pca_transformer = pca_model

        # 预计算PCA参数（如果PCA存在且显存允许）
        if pca_model is not None and memory_info['free'] > 1.0:
            self.pca_components = torch.tensor(
                pca_model.components_.T, dtype=self.dtype, device=self.device
            )
            self.pca_mean = torch.tensor(
                pca_model.mean_, dtype=self.dtype, device=self.device
            )
        else:
            self.pca_components = None
            self.pca_mean = None

    def load_feature_library(self, save_path=None):
        """加载预构建的特征库"""
        print("📂 加载内存优化特征库...")

        if save_path is None:
            save_path = self.config.save_path

        library_path = os.path.join(save_path, "memory_optimized_library.pkl")
        mapping_path = os.path.join(save_path, "identity_mapping.npy")
        pca_path = os.path.join(save_path, "pca_compressor.pkl")

        try:
            # 加载特征库
            with open(library_path, "rb") as f:
                features_cpu, labels = pickle.load(f)

            # 加载身份映射
            identity_mapping = np.load(mapping_path, allow_pickle=True).item()

            # 加载PCA
            pca_model = None
            if os.path.exists(pca_path):
                with open(pca_path, "rb") as f:
                    pca_model = pickle.load(f)

            # 轻量级加载到内存
            self._lightweight_load_to_memory(features_cpu, labels, identity_mapping, pca_model)

            print(f"✅ 特征库加载完成:")
            print(f"   库大小: {self.feature_library.shape}")
            print(f"   身份数量: {len(self.identity_mapping)}")
            print(f"   PCA状态: {'启用' if self.pca_transformer else '禁用'}")
            print(f"   设备: {self.feature_library.device}")

            return True

        except Exception as e:
            print(f"❌ 特征库加载失败: {e}")
            return False

    @torch.no_grad()
    def identify_single_image_memory_safe(self, image_path: str, threshold: float = 0.5):
        """
        内存安全的单图像识别
        """
        try:
            if not os.path.exists(image_path):
                return "Error", 0.0, 0.0, f"图像文件不存在: {image_path}"

            if self.feature_library is None:
                return "Error", 0.0, 0.0, "特征库未加载"

            start_time = self.timer.start()

            # 1. 快速图像加载
            image = cv2.imread(image_path)
            if image is None:
                return "Error", 0.0, 0.0, "图像读取失败"

            # 2. 预处理
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.fast_resize(image)

            # 3. 转换为tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
            image_tensor = image_tensor.unsqueeze(0)

            # 4. 特征提取（内存安全版本）
            query_feature = self._extract_single_feature_memory_safe(image_tensor)

            # 5. 相似度计算（处理CPU/GPU混合情况）
            if self.feature_library.device != query_feature.device:
                if self.feature_library.device.type == 'cpu':
                    # 特征库在CPU，将查询特征移到CPU
                    query_feature = query_feature.cpu()
                else:
                    # 特征库在GPU，将查询特征移到GPU
                    query_feature = query_feature.to(self.feature_library.device)

            similarity_scores = torch.matmul(query_feature, self.feature_library.T)
            max_score, max_idx = torch.max(similarity_scores, dim=1)

            latency = self.timer.end(start_time)

            # 6. 结果解析
            confidence = max_score.item()
            if confidence >= threshold:
                identity_idx = self.identity_labels[max_idx.item()]
                predicted_identity = self.identity_mapping[identity_idx]
                return predicted_identity, confidence, latency, "识别成功"
            else:
                return "Unknown", confidence, latency, f"置信度低于阈值({threshold})"

        except Exception as e:
            import traceback
            return "Error", 0.0, 0.0, f"处理异常: {str(e)}\n{traceback.format_exc()}"

    @torch.no_grad()
    def _extract_single_feature_memory_safe(self, image_tensor):
        """内存安全的单个特征提取"""
        # 移动到GPU（如果可用）
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        # 归一化
        norm_mean_gpu = self.norm_mean_cpu.to(self.device, dtype=self.dtype).view(1, 3, 1, 1)
        norm_std_gpu = self.norm_std_cpu.to(self.device, dtype=self.dtype).view(1, 3, 1, 1)
        normalized = (image_tensor - norm_mean_gpu) / norm_std_gpu

        # 特征提取
        if self.compiled_extractor is not None:
            features = self.compiled_extractor(normalized)
        else:
            features = self.model.extract_features(normalized)

        # 维度调整
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        # PCA压缩
        if self.pca_transformer is not None:
            if self.pca_components is not None and self.pca_mean is not None:
                # GPU上的快速PCA
                centered = features - self.pca_mean.unsqueeze(0)
                features = torch.matmul(centered, self.pca_components)
            else:
                # 回退到CPU PCA
                features_cpu = features.cpu().numpy()
                features_compressed = self.pca_transformer.transform(features_cpu)
                features = torch.tensor(features_compressed, dtype=self.dtype, device=self.device)

        # L2归一化
        features = F.normalize(features, p=2, dim=1)

        # 清理临时变量
        del norm_mean_gpu, norm_std_gpu, normalized

        return features


class MemoryEfficientImageDataset(Dataset):
    """内存高效的图像数据集"""

    def __init__(self, image_paths, config):
        self.image_paths = image_paths
        self.target_size = (config.img_s, config.img_s)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 使用CV2快速加载
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError("图像加载失败")

            # 快速预处理
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)

            # 转换为tensor（保持在CPU）
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)

            return tensor

        except Exception as e:
            print(f"[图像处理错误] {self.image_paths[idx]}: {e}")
            # 返回黑色图像作为fallback
            return torch.zeros(3, self.target_size[0], self.target_size[1], dtype=torch.float32)


class GPUAcceleratedTimer:
    """GPU加速计时器"""

    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'

    def start(self):
        """开始计时"""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def end(self, start_time):
        """结束计时，返回毫秒"""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start_time) * 1000


def create_memory_optimized_system(model_path=None):
    """创建内存优化的EffiLibCatReID系统实例"""
    print("🎬 创建内存优化EffiLibCatReID系统...")

    # 加载模型
    model = mobile_vit_x_small(num_classes=config.class_num)

    if model_path is None:
        model_path = os.path.join(config.save_path, f"{config.net_name}.pth")

    if not os.path.exists(model_path):
        print(f"❌ 模型权重文件不存在: {model_path}")
        return None

    print(f"📥 加载模型权重: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 兼容不同保存格式
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"✅ 加载完成，最佳准确率: {checkpoint.get('best_acc', 'Unknown')}")
            elif "model" in checkpoint:
                if hasattr(checkpoint["model"], 'state_dict'):
                    model.load_state_dict(checkpoint["model"].state_dict())
                else:
                    model = checkpoint["model"]
                print("✅ 加载完整模型对象")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 加载状态字典")
        else:
            model = checkpoint
            print("✅ 加载完整模型")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    model.to(device).eval()

    # 验证模型接口
    if not hasattr(model, 'extract_features'):
        print("❌ 模型缺少extract_features方法")
        return None

    # 创建内存优化系统
    reid_system = MemoryEfficientEffiLibCatReID(model, device, config)
    reid_system.setup_optimization_environment()

    print("🎉 内存优化EffiLibCatReID系统创建完成")
    return reid_system


def main_memory_optimized():
    """
    内存优化版主程序

    主要改进：
    - 动态批处理大小调整
    - CPU-GPU混合内存管理
    - 智能显存监控
    - 自动错误恢复
    """
    print("🚀 Memory-Optimized EffiLibCatReID: 显存友好的牛脸再识别系统")
    print("=" * 100)

    # ==================== 内存优化配置 ====================
    # 运行模式
    mode = 'batch'  # 'build', 'single', 'batch', 'interactive'

    # 路径配置
    training_data_dir = r"D:\BaiduNetdiskDownload\classification_final\data1/trainzancun"
    validation_data_dir = r"D:\BaiduNetdiskDownload\classification_final\data1/valzancun"
    single_image_path = r"D:\\BaiduNetdiskDownload\\classification_final\\datadown\\valzancun\\90249\\DSC_0242_1.jpg"

    # 内存优化参数
    base_batch_size = 16  # 起始批处理大小（会动态调整）
    pca_compression_dim = 256
    identification_threshold = 0.5
    force_rebuild = False

    # 内存管理策略
    aggressive_memory_cleanup = True  # 积极的内存清理
    cpu_fallback_enabled = True  # CPU回退模式
    max_gpu_memory_usage = 0.8  # 最大GPU内存使用率
    # ==================== 配置结束 ====================

    print(f"运行模式: {mode}")
    print(f"设备: {device}")
    print(f"基础批处理大小: {base_batch_size} (动态调整)")
    print(f"PCA压缩维度: {pca_compression_dim}")
    print(f"积极内存清理: {aggressive_memory_cleanup}")
    print("-" * 100)

    # 创建内存优化系统
    reid_system = create_memory_optimized_system()
    if reid_system is None:
        print("❌ 系统创建失败")
        return

    # 初始显存状态
    reid_system.memory_monitor.print_memory_status("系统启动")

    # 根据模式执行操作
    if mode == 'build' or force_rebuild:
        print("🏗️ 内存优化特征库构建模式")

        if not os.path.exists(training_data_dir):
            print(f"❌ 训练数据目录不存在: {training_data_dir}")
            return

        try:
            success = reid_system.build_efficient_feature_library(
                data_dir=training_data_dir,
                pca_dim=pca_compression_dim,
                base_batch_size=base_batch_size
            )

            if not success:
                print("❌ 特征库构建失败")
                return

            print("✅ 内存优化特征库构建完成")

            # 最终显存状态
            reid_system.memory_monitor.print_memory_status("构建完成")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("💥 显存不足！")
                print("🔧 建议的解决方案：")
                print("   1. 减小base_batch_size（当前：{}）".format(base_batch_size))
                print("   2. 增加pca_compression_dim以减少特征维度")
                print("   3. 设置cpu_fallback_enabled=True")
                print("   4. 关闭JIT编译（在显存极度不足时）")
                return
            else:
                raise e

    else:
        # 加载预构建特征库
        print("📂 加载预构建特征库...")
        if not reid_system.load_feature_library():
            print("❌ 特征库加载失败，请先构建特征库")
            return

    # 其他模式的执行逻辑
    if mode == 'single':
        print("🔍 单图像识别模式")

        if not os.path.exists(single_image_path):
            print(f"❌ 图像文件不存在: {single_image_path}")
            return

        print(f"\n识别图像: {os.path.basename(single_image_path)}")

        identity, confidence, latency, status = reid_system.identify_single_image_memory_safe(
            single_image_path, identification_threshold
        )

        print(f"\n{'=' * 60}")
        print(f"识别结果:")
        if identity == "Error":
            print(f"❌ 识别失败: {status}")
        elif identity == "Unknown":
            print(f"❓ 未识别出: {status}")
            print(f"   置信度: {confidence:.4f}")
        else:
            print(f"✅ 牛只身份: {identity}")
            print(f"   置信度: {confidence:.4f}")

        print(f"⏱️ 延迟: {latency:.2f} ms")
        print("=" * 60)

    elif mode == 'batch':
        print("📊 批量评估模式")

        if not os.path.exists(validation_data_dir):
            print(f"❌ 验证数据目录不存在: {validation_data_dir}")
            return

        # 批量评估的内存优化版本
        metrics = reid_system._batch_evaluate_memory_optimized(
            val_dir=validation_data_dir,
            base_batch_size=base_batch_size,
            threshold=identification_threshold
        )

    elif mode == 'interactive':
        print("🎮 交互式识别模式")
        reid_system._interactive_mode_memory_safe()

    elif mode == 'build':
        print("✅ 特征库构建完成")
        print("\n💡 后续步骤:")
        print("   1. mode='single' - 单图像测试")
        print("   2. mode='batch' - 批量评估")
        print("   3. mode='interactive' - 交互使用")

    # 最终清理
    reid_system.memory_monitor.force_cleanup()
    reid_system.memory_monitor.print_memory_status("程序结束")

    print(f"\n🎊 内存优化EffiLibCatReID运行完成！")
    print("🔧 内存优化特性:")
    print("   • 动态批处理大小调整")
    print("   • CPU-GPU混合内存管理")
    print("   • 智能显存监控与回收")
    print("   • 自动错误恢复机制")
    print("   • 分块处理大数据")


# 为MemoryEfficientEffiLibCatReID类添加缺失的方法
def add_missing_methods():
    """为内存优化类添加缺失的批量评估和交互模式方法"""

    def _batch_evaluate_memory_optimized(self, val_dir: str, base_batch_size: int = 32,
                                         threshold: float = 0.5):
        """内存优化的批量评估"""
        print("📊 开始内存优化批量评估...")

        # 收集验证数据
        val_paths, true_identities = self._collect_validation_data(val_dir)

        if len(val_paths) == 0:
            print("❌ 验证集为空")
            return None

        # 特征提取阶段计时
        feature_extract_start = time.perf_counter()
        print("提取查询特征（内存优化模式）...")
        query_features = self._extract_validation_features_memory_safe(val_paths, base_batch_size)
        feature_extract_time = time.perf_counter() - feature_extract_start

        # 分块相似度计算（避免大矩阵运算）
        print("计算相似度（分块处理）...")
        matching_start_time = time.perf_counter()
        metrics = self._compute_metrics_chunked(query_features, true_identities, threshold)
        total_matching_time = time.perf_counter() - matching_start_time

        # 计算性能指标
        num_samples = len(val_paths)
        avg_matching_time_ms = (total_matching_time * 1000) / num_samples
        total_matching_time_ms = total_matching_time * 1000
        matching_throughput = num_samples / total_matching_time

        # 添加性能指标到metrics
        metrics.update({
            'total_matching_time_ms': total_matching_time_ms,
            'avg_matching_time_ms': avg_matching_time_ms,
            'matching_throughput_fps': matching_throughput,
            'feature_extract_time_s': feature_extract_time
        })

        # 打印结果
        self._print_evaluation_results_with_timing(metrics, num_samples)

        return metrics

    def _collect_validation_data(self, val_dir):
        """收集验证数据"""
        val_paths = []
        true_identities = []

        for identity_folder in os.listdir(val_dir):
            identity_path = os.path.join(val_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            for image_file in os.listdir(identity_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    val_paths.append(os.path.join(identity_path, image_file))
                    true_identities.append(identity_folder)

        return val_paths, true_identities

    def _extract_validation_features_memory_safe(self, img_paths, base_batch_size):
        """内存安全的验证特征提取"""
        optimal_batch_size = self.memory_monitor.get_optimal_batch_size(base_batch_size)

        dataset = MemoryEfficientImageDataset(img_paths, self.config)
        dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False,
                                num_workers=min(2, psutil.cpu_count() // 2), pin_memory=True)

        all_features = []
        for batch_images in tqdm(dataloader, desc="提取验证特征"):
            try:
                batch_images = batch_images.to(self.device, dtype=self.dtype, non_blocking=True)
                features = self._extract_single_feature_memory_safe(batch_images)

                # 立即移到CPU
                all_features.append(features.cpu())

                # 清理GPU内存
                del features, batch_images

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ 验证阶段显存不足，降低批处理大小")
                    self.memory_monitor.force_cleanup()
                    optimal_batch_size = max(4, optimal_batch_size // 2)

                    # 重新创建dataloader
                    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False,
                                            num_workers=min(2, psutil.cpu_count() // 2), pin_memory=True)
                    continue
                else:
                    raise e

        self.memory_monitor.force_cleanup()
        return torch.cat(all_features, dim=0)

    def _compute_metrics_chunked(self, query_features, true_identities, threshold):
        """分块计算评估指标，避免大矩阵运算"""
        print("📊 分块计算评估指标...")

        num_queries = query_features.size(0)
        chunk_size = min(100, num_queries)  # 每次处理100个查询

        top1_correct = 0
        ap_scores = []
        predictions = []

        # 单次匹配时间统计
        individual_match_times = []

        for start_idx in tqdm(range(0, num_queries, chunk_size), desc="计算指标"):
            end_idx = min(start_idx + chunk_size, num_queries)
            query_chunk = query_features[start_idx:end_idx]

            # 移动到与特征库相同的设备
            if self.feature_library.device != query_chunk.device:
                if self.feature_library.device.type == 'cpu':
                    query_chunk = query_chunk.cpu()
                else:
                    query_chunk = query_chunk.to(self.feature_library.device)

            # 逐个查询计时（精确测量单张匹配时间）
            for i in range(query_chunk.size(0)):
                single_query = query_chunk[i:i + 1]

                # 单次匹配计时
                single_match_start = self.timer.start()
                similarity_scores = torch.matmul(single_query, self.feature_library.T)
                max_score, max_idx = torch.max(similarity_scores, dim=1)
                single_match_time = self.timer.end(single_match_start)

                individual_match_times.append(single_match_time)

                # 处理结果
                q_idx = start_idx + i
                true_identity = true_identities[q_idx]

                # Top-1准确率
                best_match_idx = max_idx.item()
                predicted_identity_idx = self.identity_labels[best_match_idx]
                predicted_identity = self.identity_mapping[predicted_identity_idx]

                if predicted_identity == true_identity:
                    top1_correct += 1

                # 阈值预测
                confidence = max_score.item()
                if confidence >= threshold:
                    predictions.append(predicted_identity)
                else:
                    predictions.append("Unknown")

                # mAP计算（简化版本）
                scores = similarity_scores.squeeze().cpu().numpy()
                y_true = np.array([
                    1 if self.identity_mapping[self.identity_labels[idx]] == true_identity else 0
                    for idx in range(len(self.identity_labels))
                ])

                if np.sum(y_true) > 0:
                    try:
                        ap = average_precision_score(y_true, scores)
                        ap_scores.append(ap)
                    except:
                        pass  # 跳过有问题的样本

            # 清理当前块
            del query_chunk

        # 计算匹配时间统计
        avg_match_time_ms = np.mean(individual_match_times)
        min_match_time_ms = np.min(individual_match_times)
        max_match_time_ms = np.max(individual_match_times)
        std_match_time_ms = np.std(individual_match_times)
        p95_match_time_ms = np.percentile(individual_match_times, 95)
        p99_match_time_ms = np.percentile(individual_match_times, 99)

        # 计算最终指标
        top1_accuracy = top1_correct / num_queries
        mAP = np.mean(ap_scores) if ap_scores else 0.0

        # 计算识别率
        recognized_indices = [i for i, pred in enumerate(predictions) if pred != "Unknown"]
        recognition_rate = len(recognized_indices) / num_queries

        # 计算传统指标
        if len(recognized_indices) > 0:
            y_true_rec = [true_identities[i] for i in recognized_indices]
            y_pred_rec = [predictions[i] for i in recognized_indices]

            accuracy = accuracy_score(y_true_rec, y_pred_rec)
            precision = precision_score(y_true_rec, y_pred_rec, average='weighted', zero_division=0)
            recall = recall_score(y_true_rec, y_pred_rec, average='weighted', zero_division=0)
            f1 = f1_score(y_true_rec, y_pred_rec, average='weighted', zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0

        return {
            'top1_accuracy': top1_accuracy,
            'mAP': mAP,
            'recognition_rate': recognition_rate,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'recognized_count': len(recognized_indices),
            'total_queries': num_queries,
            # 新增的匹配时间统计
            'avg_match_time_ms': avg_match_time_ms,
            'min_match_time_ms': min_match_time_ms,
            'max_match_time_ms': max_match_time_ms,
            'std_match_time_ms': std_match_time_ms,
            'p95_match_time_ms': p95_match_time_ms,
            'p99_match_time_ms': p99_match_time_ms,
            'individual_match_times': individual_match_times
        }

    def _print_evaluation_results_with_timing(self, metrics, total_samples):
        """包含详细时间统计的评估结果打印"""
        print(f"\n{'=' * 80}")
        print(f"内存优化 EffiLibCatReID 评估结果")
        print(f"{'=' * 80}")

        # 准确率指标
        print(f"📊 准确率指标:")
        print(f"   Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy'] * 100:.2f}%)")
        print(f"   mAP (Mean Average Precision): {metrics['mAP']:.4f} ({metrics['mAP'] * 100:.2f}%)")
        print(
            f"   识别率: {metrics['recognition_rate']:.4f} ({metrics['recognized_count']}/{metrics['total_queries']})")
        print(f"   准确率: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"   精确率: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
        print(f"   召回率: {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
        print(f"   F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")

        # 性能指标 - 重点突出匹配时间
        print(f"\n⚡ 匹配性能指标:")
        print(f"   📈 总匹配时间: {metrics.get('total_matching_time_ms', 0):.2f} ms")
        print(f"   🎯 单张匹配时间: {metrics.get('avg_matching_time_ms', 0):.2f} ms")
        print(f"   🚀 匹配吞吐量: {metrics.get('matching_throughput_fps', 0):.1f} fps")
        print(f"   📊 最快匹配: {metrics.get('min_match_time_ms', 0):.2f} ms")
        print(f"   📊 最慢匹配: {metrics.get('max_match_time_ms', 0):.2f} ms")
        print(f"   📊 时间标准差: {metrics.get('std_match_time_ms', 0):.2f} ms")
        print(f"   📊 P95延迟: {metrics.get('p95_match_time_ms', 0):.2f} ms")
        print(f"   📊 P99延迟: {metrics.get('p99_match_time_ms', 0):.2f} ms")

        # 其他性能指标
        if 'feature_extract_time_s' in metrics:
            extract_time = metrics['feature_extract_time_s']
            avg_extract_time = (extract_time * 1000) / total_samples
            print(f"\n🔧 特征提取性能:")
            print(f"   总提取时间: {extract_time:.2f} s")
            print(f"   单张提取时间: {avg_extract_time:.2f} ms")
            print(f"   提取吞吐量: {total_samples / extract_time:.1f} fps")

        # 性能等级评估
        avg_match_time = metrics.get('avg_matching_time_ms', float('inf'))
        print(f"\n🏆 性能等级评估:")
        if avg_match_time < 3:
            print(f"🚀 匹配性能: 超高速 (<3ms) - 超越EffiLibCatReID设计目标!")
        elif avg_match_time < 5:
            print(f"⚡ 匹配性能: 极速 (<5ms) - 达到EffiLibCatReID设计目标")
        elif avg_match_time < 10:
            print(f"✅ 匹配性能: 高速 (<10ms) - 符合实时应用要求")
        elif avg_match_time < 20:
            print(f"✅ 匹配性能: 中速 (<20ms) - 可用于实际部署")
        else:
            print(f"⚠️ 匹配性能: 需优化 (>{avg_match_time:.1f}ms)")

        # 延迟稳定性评估
        std_time = metrics.get('std_match_time_ms', 0)
        if std_time < 1:
            print(f"📈 延迟稳定性: 极佳 (标准差<1ms)")
        elif std_time < 2:
            print(f"📈 延迟稳定性: 良好 (标准差<2ms)")
        else:
            print(f"📈 延迟稳定性: 需改进 (标准差{std_time:.1f}ms)")

        print(f"\n📋 测试概况:")
        print(f"   总样本数: {total_samples}")
        print(f"   特征库大小: {self.feature_library.shape}")
        print(f"   运行设备: {self.device}")
        print(f"{'=' * 80}")

        # 如果有详细时间数据，提供进一步分析
        if 'individual_match_times' in metrics:
            times = metrics['individual_match_times']
            under_5ms = sum(1 for t in times if t < 5) / len(times) * 100
            under_10ms = sum(1 for t in times if t < 10) / len(times) * 100

            print(f"\n📊 延迟达标率分析:")
            print(f"   <5ms达标率: {under_5ms:.1f}%")
            print(f"   <10ms达标率: {under_10ms:.1f}%")
            print(f"{'=' * 80}")

    def _interactive_mode_memory_safe(self):
        """内存安全的交互式模式"""
        print(f"\n{'=' * 80}")
        print(f"🐄 内存优化 EffiLibCatReID 交互式识别")
        print(f"{'=' * 80}")
        print("输入图像路径进行识别，输入 'quit' 退出")
        print("输入 'memory' 查看显存状态")
        print("输入 'cleanup' 强制清理显存")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n请输入命令: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("退出内存优化识别系统")
                    break

                if user_input.lower() == 'memory':
                    self.memory_monitor.print_memory_status("用户查询")
                    continue

                if user_input.lower() == 'cleanup':
                    print("🧹 执行显存清理...")
                    self.memory_monitor.force_cleanup()
                    self.memory_monitor.print_memory_status("清理后")
                    continue

                if not user_input:
                    continue

                image_path = user_input.strip('\'"')
                print(f"\n🔍 识别: {os.path.basename(image_path)}")

                identity, confidence, latency, status = self.identify_single_image_memory_safe(
                    image_path, 0.5
                )

                # 显示结果
                if identity == "Error":
                    print(f"❌ 识别失败: {status}")
                elif identity == "Unknown":
                    print(f"❓ 未识别出: {status}")
                    print(f"   置信度: {confidence:.4f}")
                else:
                    print(f"✅ 牛只身份: {identity}")
                    print(f"   置信度: {confidence:.4f}")

                print(f"⏱️ 延迟: {latency:.2f} ms")

                # 如果显存允许，显示延迟性能评估
                if latency < 3:
                    print(f"🚀 匹配速度: 超高速 (<3ms)")
                elif latency < 5:
                    print(f"⚡ 匹配速度: 极速 (<5ms)")
                elif latency < 10:
                    print(f"✅ 匹配速度: 高速 (<10ms)")
                else:
                    print(f"⚠️ 匹配速度: 需优化 ({latency:.1f}ms)")

            except KeyboardInterrupt:
                print("\n\n用户中断，退出系统")
                break
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                self.memory_monitor.force_cleanup()

    # 将方法绑定到类
    MemoryEfficientEffiLibCatReID._batch_evaluate_memory_optimized = _batch_evaluate_memory_optimized
    MemoryEfficientEffiLibCatReID._collect_validation_data = _collect_validation_data
    MemoryEfficientEffiLibCatReID._extract_validation_features_memory_safe = _extract_validation_features_memory_safe
    MemoryEfficientEffiLibCatReID._compute_metrics_chunked = _compute_metrics_chunked
    MemoryEfficientEffiLibCatReID._print_evaluation_results_with_timing = _print_evaluation_results_with_timing
    MemoryEfficientEffiLibCatReID._interactive_mode_memory_safe = _interactive_mode_memory_safe


# 显存使用建议和故障排除
def print_memory_optimization_tips():
    """打印显存优化建议"""
    print("\n🧠 显存优化建议:")
    print("=" * 60)
    print("🔧 参数调优:")
    print("   • base_batch_size: 16 -> 8 -> 4 (显存不足时)")
    print("   • pca_compression_dim: 256 -> 128 -> 64")
    print("   • 禁用JIT编译: compiled_extractor = None")
    print()
    print("⚡ 运行时优化:")
    print("   • 启用aggressive_memory_cleanup = True")
    print("   • 使用cpu_fallback_enabled = True")
    print("   • 定期执行memory cleanup")
    print()
    print("🎯 硬件建议:")
    print("   • 推荐显存: 6GB+ (RTX 3060及以上)")
    print("   • 最低显存: 4GB (GTX 1650及以上)")
    print("   • 备选方案: CPU模式 (device='cpu')")
    print("=" * 60)


if __name__ == "__main__":
    # 添加缺失的方法
    add_missing_methods()

    # 打印优化建议
    print_memory_optimization_tips()

    # 运行主程序
    main_memory_optimized()
