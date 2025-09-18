#!/usr/bin/env python3
"""
Pocket Detection Inference Script
用于对长音频文件进行pocket检测的推理脚本
"""

import os
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from supervised_model import AudioCNNLSTMClassifier

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 确保数值类型正确转换
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['model']['num_classes'] = int(config['model']['num_classes'])
    config['model']['dropout_rate'] = float(config['model']['dropout_rate'])
    config['data']['n_mels'] = int(config['data']['n_mels'])
    config['data']['frames'] = int(config['data']['frames'])
    config['data']['sample_rate'] = int(config['data']['sample_rate'])
    config['data']['n_fft'] = int(config['data']['n_fft'])
    config['data']['window_duration'] = float(config['data']['window_duration'])
    config['data']['num_frames'] = int(config['data']['num_frames'])
    config['data']['hop_length'] = float(config['data']['hop_length'])
    
    return config

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def extract_mel_spectrogram(signal, sr, n_fft, n_mels, target_shape=(128, 128)):
    """提取mel频谱图特征"""
    # 计算mel频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )
    
    # 转换为log scale
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # 标准化
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
    
    # 转置为 (time, freq)
    features = log_mel_spec.T
    
    # 调整大小到目标形状
    import cv2
    features_resized = cv2.resize(features.astype(np.float32), target_shape, interpolation=cv2.INTER_LINEAR)
    
    # 确保输出形状正确
    if features_resized.shape != target_shape:
        h, w = features_resized.shape
        target_h, target_w = target_shape
        features_fixed = np.zeros(target_shape, dtype=np.float32)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        features_fixed[:copy_h, :copy_w] = features_resized[:copy_h, :copy_w]
        features_resized = features_fixed
    
    return features_resized

def extract_consecutive_frames_from_signal(signal, sr, config, start_sample, window_samples):
    """从长音频信号中提取5个连续帧"""
    try:
        # 提取窗口信号
        window_signal = signal[start_sample:start_sample + window_samples]
        
        # Calculate parameters for 5 frames from 1-second audio
        window_duration = config['data']['window_duration']  # 1.0 second
        num_frames = config['data']['num_frames']  # 5 frames
        hop_length = config['data']['hop_length']  # 0.2 second
        
        frame_duration = window_duration / num_frames  # 0.2 second per frame
        frame_samples = int(frame_duration * sr)
        hop_samples = int(hop_length * sr)
        
        frames = []
        
        # Extract 5 consecutive frames from the 1-second window
        for i in range(num_frames):
            frame_start = i * hop_samples
            frame_end = frame_start + frame_samples
            
            # Ensure we don't exceed window length
            if frame_end > len(window_signal):
                # Pad with zeros if necessary
                frame_signal = np.zeros(frame_samples)
                available_samples = len(window_signal) - frame_start
                if available_samples > 0:
                    frame_signal[:available_samples] = window_signal[frame_start:]
            else:
                frame_signal = window_signal[frame_start:frame_end]
            
            # Extract mel-spectrogram for this frame
            frame_features = extract_mel_spectrogram(
                frame_signal, sr, 
                config['data']['n_fft'], 
                config['data']['n_mels']
            )
            frames.append(frame_features)
        
        # Stack frames: (num_frames, height, width)
        consecutive_frames = np.stack(frames, axis=0)
        return consecutive_frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        # Return zero frames in case of error
        num_frames = config['data']['num_frames']
        return np.zeros((num_frames, 128, 128), dtype=np.float32)

def load_model(model_path, config, device):
    """加载训练好的模型"""
    model = AudioCNNLSTMClassifier(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        num_frames=config['data']['num_frames']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    best_val_acc = checkpoint.get('best_val_acc', None)
    if best_val_acc is not None:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    else:
        print("Best validation accuracy: N/A")
    
    return model

def process_long_audio(audio_path, model, config, device, window_length=1.0, overlap=0.5, threshold=0.5):
    """
    处理长音频文件，使用滑动窗口进行检测
    
    Args:
        audio_path: 音频文件路径
        model: 训练好的模型
        config: 配置文件
        device: 设备
        window_length: 窗口长度（秒）- 现在使用1秒窗口
        overlap: 重叠比例 (0-1)
        threshold: 分类阈值
    
    Returns:
        results: 检测结果列表
    """
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    
    # 加载音频文件（保持原始采样率）
    signal, original_sr = librosa.load(audio_path, sr=None)
    total_duration = len(signal) / original_sr
    print(f"Original sample rate: {original_sr} Hz")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # 重采样到模型要求的采样率
    target_sr = config['data']['sample_rate']
    if original_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)
        print(f"Resampled to: {target_sr} Hz")
    
    # 计算窗口参数
    window_samples = int(window_length * target_sr)
    step_samples = int(window_samples * (1 - overlap))
    
    results = []
    anomaly_segments = []  # 存储leak和pocket段
    
    # 滑动窗口处理
    total_windows = (len(signal) - window_samples) // step_samples + 1
    print(f"Processing {total_windows} windows...")
    
    # 类别名称映射
    class_names = {0: 'Normal', 1: 'Leak', 2: 'Pocket'}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(signal) - window_samples + 1, step_samples), desc="Analyzing"):
            # 计算时间戳
            start_time = i / target_sr
            end_time = (i + window_samples) / target_sr
            
            try:
                # 提取5个连续帧特征
                consecutive_frames = extract_consecutive_frames_from_signal(
                    signal, target_sr, config, i, window_samples
                )
                
                # 转换为tensor并添加batch维度: (num_frames, H, W) -> (1, num_frames, 1, H, W)
                features_tensor = torch.FloatTensor(consecutive_frames).unsqueeze(0).unsqueeze(2).to(device)
                
                # 模型预测
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 获取最高概率的类别
                predicted_class = torch.argmax(probabilities, dim=1).item()
                max_prob = probabilities[0][int(predicted_class)].item()
                
                # 检查是否为异常（leak或pocket）
                is_anomaly = predicted_class in [1, 2] and max_prob > threshold
                
                result = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_time_str': str(timedelta(seconds=int(start_time))),
                    'end_time_str': str(timedelta(seconds=int(end_time))),
                    'predicted_class': predicted_class,
                    'predicted_class_name': class_names[int(predicted_class)],
                    'confidence': max_prob,
                    'normal_prob': probabilities[0, 0].item(),
                    'leak_prob': probabilities[0, 1].item(),
                    'pocket_prob': probabilities[0, 2].item(),
                    'is_anomaly': is_anomaly,
                    'features': consecutive_frames  # 保存5帧特征用于可视化
                }
                
                results.append(result)
                
                if is_anomaly:
                    anomaly_segments.append(result)
                    
            except Exception as e:
                print(f"Error processing window at {start_time:.2f}s: {e}")
                continue
    
    return results, anomaly_segments

def merge_consecutive_detections(anomaly_segments, max_gap=2.0):
    """合并连续的异常检测（leak或pocket）"""
    if not anomaly_segments:
        return []
    
    # 按类别分组
    leak_segments = [seg for seg in anomaly_segments if seg['predicted_class'] == 1]
    pocket_segments = [seg for seg in anomaly_segments if seg['predicted_class'] == 2]
    
    def merge_segments_by_class(segments, class_name):
        if not segments:
            return []
        
        merged = []
        current_start = segments[0]['start_time']
        current_end = segments[0]['end_time']
        current_max_prob = segments[0]['confidence']
        
        for segment in segments[1:]:
            # 如果当前段与前一段的间隔小于max_gap，则合并
            if segment['start_time'] - current_end <= max_gap:
                current_end = segment['end_time']
                current_max_prob = max(current_max_prob, segment['confidence'])
            else:
                # 保存当前合并的段
                merged.append({
                    'start_time': current_start,
                    'end_time': current_end,
                    'start_time_str': str(timedelta(seconds=int(current_start))),
                    'end_time_str': str(timedelta(seconds=int(current_end))),
                    'duration': current_end - current_start,
                    'max_confidence': current_max_prob,
                    'class_name': class_name
                })
                
                # 开始新的段
                current_start = segment['start_time']
                current_end = segment['end_time']
                current_max_prob = segment['confidence']
        
        # 添加最后一段
        merged.append({
            'start_time': current_start,
            'end_time': current_end,
            'start_time_str': str(timedelta(seconds=int(current_start))),
            'end_time_str': str(timedelta(seconds=int(current_end))),
            'duration': current_end - current_start,
            'max_confidence': current_max_prob,
            'class_name': class_name
        })
        
        return merged
    
    # 分别合并leak和pocket段
    merged_leaks = merge_segments_by_class(leak_segments, 'Leak')
    merged_pockets = merge_segments_by_class(pocket_segments, 'Pocket')
    
    # 合并所有段并按时间排序
    all_merged = merged_leaks + merged_pockets
    all_merged.sort(key=lambda x: x['start_time'])
    
    return all_merged

def create_spectrogram_visualization(audio_path, all_results, merged_segments, config, output_dir):
    """创建带有检测框的时频谱图可视化"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(audio_path).replace('.WAV', '')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频文件
    signal, original_sr = librosa.load(audio_path, sr=None)
    target_sr = config['data']['sample_rate']
    if original_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)
    
    # 计算完整音频的mel频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=target_sr,
        n_fft=config['data']['n_fft'],
        n_mels=config['data']['n_mels']
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # 创建时间轴（分钟）
    time_axis = librosa.frames_to_time(
        range(log_mel_spec.shape[1]), 
        sr=target_sr, 
        n_fft=config['data']['n_fft']
    ) / 60
    
    # 创建频率轴
    freq_axis = librosa.mel_frequencies(n_mels=config['data']['n_mels'])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # 上图：完整的mel频谱图
    im1 = ax1.imshow(
        log_mel_spec, 
        aspect='auto', 
        origin='lower',
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        cmap='viridis'
    )
    
    ax1.set_title(f'Mel Spectrogram with Anomaly Detection - {filename}', fontsize=16)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Frequency (Hz)')
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Power (dB)')
    
    # 在频谱图上标记检测到的异常段
    colors = {'Leak': 'red', 'Pocket': 'orange'}
    for seg in merged_segments:
        start_min = seg['start_time'] / 60
        end_min = seg['end_time'] / 60
        class_name = seg['class_name']
        confidence = seg['max_confidence']
        
        rect = Rectangle(
            (start_min, freq_axis[0]), 
            end_min - start_min, 
            freq_axis[-1] - freq_axis[0],
            linewidth=3, 
            edgecolor=colors[class_name], 
            facecolor='none',
            alpha=0.8
        )
        ax1.add_patch(rect)
        ax1.add_patch(rect)
        
        # 添加标注
        mid_time = (start_min + end_min) / 2
        mid_freq = (freq_axis[0] + freq_axis[-1]) / 2
        
        ax1.annotate(
            f'{class_name}\n{seg["start_time_str"]}-{seg["end_time_str"]}\nConf: {confidence:.3f}',
            xy=(mid_time, mid_freq),
            xytext=(10, 10), 
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[class_name], alpha=0.7),
            fontsize=10,
            color='white',
            weight='bold',
            ha='center'
        )
    
    # 下图：概率时间线
    times = [r['start_time'] / 60 for r in all_results]
    normal_probs = [r['normal_prob'] for r in all_results]
    leak_probs = [r['leak_prob'] for r in all_results]
    pocket_probs = [r['pocket_prob'] for r in all_results]
    
    ax2.plot(times, normal_probs, label='Normal', color='green', alpha=0.7, linewidth=1)
    ax2.plot(times, leak_probs, label='Leak', color='red', alpha=0.7, linewidth=1)
    ax2.plot(times, pocket_probs, label='Pocket', color='orange', alpha=0.7, linewidth=1)
    
    # 标记检测到的异常段
    legend_added = {'Leak': False, 'Pocket': False}
    for seg in merged_segments:
        start_min = seg['start_time'] / 60
        end_min = seg['end_time'] / 60
        class_name = seg['class_name']
        
        label = f'{class_name} Detection' if not legend_added[class_name] else None
        ax2.axvspan(start_min, end_min, alpha=0.3, color=colors[class_name], label=label)
        legend_added[class_name] = True
    
    ax2.set_title('Classification Probabilities Over Time')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存图表
    plot_filename = f"{output_dir}/spectrogram_detection_{filename}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def save_results(file_results, output_dir):
    """保存检测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    summary_data = []
    
    for filename, (all_results, anomaly_segments, merged_segments) in file_results.items():
        # 统计信息
        total_windows = len(all_results)
        leak_windows = len([r for r in all_results if r['predicted_class'] == 1])
        pocket_windows = len([r for r in all_results if r['predicted_class'] == 2])
        anomaly_windows = leak_windows + pocket_windows
        
        # 分别统计leak和pocket段
        leak_segments = [seg for seg in merged_segments if seg['class_name'] == 'Leak']
        pocket_segments = [seg for seg in merged_segments if seg['class_name'] == 'Pocket']
        
        # 计算总的异常时长
        total_leak_duration = sum(seg['duration'] for seg in leak_segments)
        total_pocket_duration = sum(seg['duration'] for seg in pocket_segments)
        total_anomaly_duration = total_leak_duration + total_pocket_duration
        
        summary_data.append({
            'filename': filename,
            'total_windows': total_windows,
            'leak_windows': leak_windows,
            'pocket_windows': pocket_windows,
            'anomaly_windows': anomaly_windows,
            'anomaly_ratio': anomaly_windows / total_windows if total_windows > 0 else 0,
            'leak_segments_count': len(leak_segments),
            'pocket_segments_count': len(pocket_segments),
            'total_leak_duration_seconds': total_leak_duration,
            'total_pocket_duration_seconds': total_pocket_duration,
            'total_anomaly_duration_seconds': total_anomaly_duration,
            'total_anomaly_duration_minutes': total_anomaly_duration / 60
        })
        
        # 保存每个文件的详细结果
        detail_filename = f"{output_dir}/detailed_results_{filename.replace('.WAV', '')}_{timestamp}.csv"
        df_detail = pd.DataFrame(all_results)
        # 移除features列（太大了）
        if 'features' in df_detail.columns:
            df_detail = df_detail.drop('features', axis=1)
        df_detail.to_csv(detail_filename, index=False)
        
        # 保存合并后的异常段
        if merged_segments:
            anomaly_filename = f"{output_dir}/anomaly_segments_{filename.replace('.WAV', '')}_{timestamp}.csv"
            df_anomaly = pd.DataFrame(merged_segments)
            df_anomaly.to_csv(anomaly_filename, index=False)
    
    # 保存汇总结果
    summary_filename = f"{output_dir}/detection_summary_{timestamp}.csv"
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_filename, index=False)
    
    return summary_filename, df_summary

def create_visualization(file_results, output_dir, threshold=0.5):
    """创建可视化图表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Leak and Pocket Detection Analysis Results', fontsize=16)
    
    # 准备数据
    filenames = []
    leak_ratios = []
    pocket_ratios = []
    leak_counts = []
    pocket_counts = []
    leak_durations = []
    pocket_durations = []
    
    for filename, (all_results, anomaly_segments, merged_segments) in file_results.items():
        filenames.append(filename.replace('.WAV', ''))
        
        total_windows = len(all_results)
        leak_windows = len([r for r in all_results if r['predicted_class'] == 1])
        pocket_windows = len([r for r in all_results if r['predicted_class'] == 2])
        
        leak_ratio = leak_windows / total_windows if total_windows > 0 else 0
        pocket_ratio = pocket_windows / total_windows if total_windows > 0 else 0
        leak_ratios.append(leak_ratio * 100)
        pocket_ratios.append(pocket_ratio * 100)
        
        leak_segments = [seg for seg in merged_segments if seg['class_name'] == 'Leak']
        pocket_segments = [seg for seg in merged_segments if seg['class_name'] == 'Pocket']
        
        leak_counts.append(len(leak_segments))
        pocket_counts.append(len(pocket_segments))
        
        total_leak_duration = sum(seg['duration'] for seg in leak_segments)
        total_pocket_duration = sum(seg['duration'] for seg in pocket_segments)
        leak_durations.append(total_leak_duration / 60)  # 转换为分钟
        pocket_durations.append(total_pocket_duration / 60)
    
    # 图1: Leak检测比例
    axes[0, 0].bar(filenames, leak_ratios, color='red', alpha=0.7)
    axes[0, 0].set_title('Leak Detection Ratio (%)')
    axes[0, 0].set_ylabel('Percentage of Windows with Leak')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 图2: Pocket检测比例
    axes[0, 1].bar(filenames, pocket_ratios, color='orange', alpha=0.7)
    axes[0, 1].set_title('Pocket Detection Ratio (%)')
    axes[0, 1].set_ylabel('Percentage of Windows with Pocket')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 图3: 异常段数量对比
    x = np.arange(len(filenames))
    width = 0.35
    axes[0, 2].bar(x - width/2, leak_counts, width, label='Leak', color='red', alpha=0.7)
    axes[0, 2].bar(x + width/2, pocket_counts, width, label='Pocket', color='orange', alpha=0.7)
    axes[0, 2].set_title('Number of Anomaly Segments')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(filenames, rotation=45)
    axes[0, 2].legend()
    
    # 图4: Leak时长
    axes[1, 0].bar(filenames, leak_durations, color='red', alpha=0.7)
    axes[1, 0].set_title('Total Leak Duration (Minutes)')
    axes[1, 0].set_ylabel('Duration (Minutes)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 图5: Pocket时长
    axes[1, 1].bar(filenames, pocket_durations, color='orange', alpha=0.7)
    axes[1, 1].set_title('Total Pocket Duration (Minutes)')
    axes[1, 1].set_ylabel('Duration (Minutes)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 图6: 概率分布热图（选择第一个文件作为示例）
    if file_results:
        first_file = list(file_results.keys())[0]
        all_results, _, merged_segments = file_results[first_file]
        
        # 创建概率分布热图
        times = [r['start_time'] / 60 for r in all_results]  # 转换为分钟
        normal_probs = [r['normal_prob'] for r in all_results]
        leak_probs = [r['leak_prob'] for r in all_results]
        pocket_probs = [r['pocket_prob'] for r in all_results]
        
        # 堆叠条形图显示概率分布
        axes[1, 2].plot(times, normal_probs, label='Normal', color='green', alpha=0.7)
        axes[1, 2].plot(times, leak_probs, label='Leak', color='red', alpha=0.7)
        axes[1, 2].plot(times, pocket_probs, label='Pocket', color='orange', alpha=0.7)
        axes[1, 2].axhline(y=threshold, color='black', linestyle='--', alpha=0.7, label='Threshold')
        
        # 标记检测到的异常段
        for seg in merged_segments:
            start_min = seg['start_time'] / 60
            end_min = seg['end_time'] / 60
            color = 'red' if seg['class_name'] == 'Leak' else 'orange'
            axes[1, 2].axvspan(start_min, end_min, alpha=0.3, color=color)
        
        axes[1, 2].set_title(f'Classification Probabilities - {first_file.replace(".WAV", "")}')
        axes[1, 2].set_xlabel('Time (Minutes)')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存图表
    plot_filename = f"{output_dir}/detection_visualization_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def main():
    """主函数"""
    # 配置路径
    config_path = "./supervised_config.yaml"
    model_path = "./checkpoints/best_model.pth"
    # audio_dir = r"..\..\raw_leak_pocket\interference\Project-14-Portland"
    # audio_dir = r"..\..\raw_leak_pocket\interference\Project-7-Alex"  # pocket
    audio_dir = r"..\..\raw_leak_pocket\interference\Project-1"   # leak
    output_dir = "./inference_results"
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    # 加载配置和模型
    config = load_config(config_path)
    device = setup_device()
    model = load_model(model_path, config, device)
    
    # 获取音频文件列表
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.WAV')]
    audio_files.sort()
    
    print(f"\nFound {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  - {f}")
    
    # 推理参数 - 与训练数据创建参数保持一致
    window_length = 1.0  # 1秒窗口（匹配新的window_duration）
    overlap = 0.50       # 50%重叠
    threshold = 0.395      # 分类阈值（降低阈值以检测更多异常）
    
    print(f"\nInference parameters:")
    print(f"  Window length: {window_length} seconds")
    print(f"  Overlap: {overlap * 100}%")
    print(f"  Classification threshold: {threshold}")
    print(f"  Using {config['data']['num_frames']} consecutive frames per window")
    
    # 处理每个音频文件
    file_results = {}
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        
        # 处理音频
        all_results, anomaly_segments = process_long_audio(
            audio_path, model, config, device, 
            window_length=window_length, 
            overlap=overlap, 
            threshold=threshold
        )
        
        # 合并连续检测
        merged_segments = merge_consecutive_detections(anomaly_segments, max_gap=2.0)
        
        # 创建时频谱图可视化
        if merged_segments:  # 只有检测到异常时才创建时频谱图
            spectrogram_file = create_spectrogram_visualization(
                audio_path, all_results, merged_segments, config, output_dir
            )
            print(f"  Spectrogram saved: {os.path.basename(spectrogram_file)}")
        
        # 保存结果
        file_results[audio_file] = (all_results, anomaly_segments, merged_segments)
        
        # 打印简要结果
        total_windows = len(all_results)
        leak_windows = len([r for r in all_results if r['predicted_class'] == 1])
        pocket_windows = len([r for r in all_results if r['predicted_class'] == 2])
        anomaly_windows = leak_windows + pocket_windows
        
        leak_segments = [seg for seg in merged_segments if seg['class_name'] == 'Leak']
        pocket_segments = [seg for seg in merged_segments if seg['class_name'] == 'Pocket']
        
        total_leak_duration = sum(seg['duration'] for seg in leak_segments)
        total_pocket_duration = sum(seg['duration'] for seg in pocket_segments)
        total_anomaly_duration = total_leak_duration + total_pocket_duration
        
        print(f"\nResults for {audio_file}:")
        print(f"  Total analysis windows: {total_windows}")
        print(f"  Windows with leak detected: {leak_windows} ({leak_windows/total_windows*100:.1f}%)")
        print(f"  Windows with pocket detected: {pocket_windows} ({pocket_windows/total_windows*100:.1f}%)")
        print(f"  Total anomaly windows: {anomaly_windows} ({anomaly_windows/total_windows*100:.1f}%)")
        print(f"  Number of leak segments: {len(leak_segments)}")
        print(f"  Number of pocket segments: {len(pocket_segments)}")
        print(f"  Total leak duration: {total_leak_duration:.1f} seconds ({total_leak_duration/60:.1f} minutes)")
        print(f"  Total pocket duration: {total_pocket_duration:.1f} seconds ({total_pocket_duration/60:.1f} minutes)")
        print(f"  Total anomaly duration: {total_anomaly_duration:.1f} seconds ({total_anomaly_duration/60:.1f} minutes)")
        
        if merged_segments:
            print(f"  Detected anomaly segments:")
            for i, seg in enumerate(merged_segments, 1):
                print(f"    {i}: {seg['class_name']} - {seg['start_time_str']} to {seg['end_time_str']} "
                      f"(duration: {seg['duration']:.1f}s, confidence: {seg['max_confidence']:.3f})")
        else:
            print(f"  No anomaly segments detected.")
    
    # 保存所有结果
    summary_filename, df_summary = save_results(file_results, output_dir)
    plot_filename = create_visualization(file_results, output_dir, threshold)
    
    print(f"\n" + "="*80)
    print("LEAK AND POCKET DETECTION INFERENCE COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_filename}")
    print(f"Visualization: {plot_filename}")
    
    # 打印总体统计
    print(f"\nOverall Statistics:")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        print(f"File: {row['filename']}")
        print(f"  Total windows: {row['total_windows']}")
        print(f"  Leak detection: {row['leak_windows']} windows ({row['leak_windows']/row['total_windows']*100:.1f}%), "
              f"{row['leak_segments_count']} segments, {row['total_leak_duration_seconds']:.1f}s")
        print(f"  Pocket detection: {row['pocket_windows']} windows ({row['pocket_windows']/row['total_windows']*100:.1f}%), "
              f"{row['pocket_segments_count']} segments, {row['total_pocket_duration_seconds']:.1f}s")
        print(f"  Total anomaly: {row['anomaly_windows']} windows ({row['anomaly_ratio']*100:.1f}%), "
              f"{row['total_anomaly_duration_minutes']:.1f} minutes")
        print("-" * 80)

if __name__ == "__main__":
    main()
