#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

from lpm.model import RiskNet
from lpm.data.datasets.risknet import (
    RiskNetBatchGenerator,
    TrajectoryValidator,
)
from lpm.data.datasets.risknet.dataset import load_test_dataset
from lpm.model.risknet.utils import get_predictions_and_labels

def analyze_precision_recall_at_thresholds(y_true, y_pred, start_threshold=0.001, end_threshold=1.0, step=0.001):
    thresholds = np.arange(start_threshold, end_threshold, step)
    results = []
    
    print(f"分析 {len(thresholds)} 个阈值...")
    
    for threshold in tqdm(thresholds, desc="分析阈值", unit="threshold"):
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        precision = skm.precision_score(y_true, y_pred_binary, zero_division=0)
        recall = skm.recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = skm.f1_score(y_true, y_pred_binary, zero_division=0)
        
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred_binary, labels=[0,1]).ravel()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn), 
            'fn': int(fn),
            'total_predicted_positive': int(tp + fp),
            'total_actual_positive': int(tp + fn)
        })
    
    return pd.DataFrame(results)

def main():
    model_path = "outputs/transformer_1k_v2/risknet.keras"
    
    print(f"加载模型: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        for alt_path in ["outputs/transformer/risknet.keras", "outputs/transformer_500/risknet.keras"]:
            if Path(alt_path).exists():
                print(f"尝试加载: {alt_path}")
                model = tf.keras.models.load_model(alt_path)
                model_path = alt_path
                break
    
    from omegaconf import OmegaConf
    data_config = OmegaConf.load("config/risknet/data/default.yaml")
    
    test_data_path = Path("sps_data_1k_v2")
    print(f"使用测试数据: {test_data_path}")
    
    print("加载实际测试数据...")
    trajectory_validator = TrajectoryValidator(data_config)
    test_dataset = load_test_dataset(test_data_path, trajectory_validator, num_processes=4)
    
    print(f"测试数据统计:")
    print(f"  正样本数: {test_dataset.pos.n_pos}")
    print(f"  负样本数: {test_dataset.neg.n_neg}")
    
    print("创建批生成器...")
    test_batch_generator = RiskNetBatchGenerator(
        batch_size=64,
        max_codes=model.input_dim,
        tokenizer=model.tokenizer,
        n_trajectories=10
    )
    
    test_seq = test_batch_generator.flow(test_dataset.pos, test_dataset.neg, shuffle=False)
    
    print("获取模型预测...")
    print(f"总共需要处理 {len(test_seq)} 个批次...")
    y_true, y_pred_logits = get_predictions_and_labels(model, test_seq, verbose=1)
    
    y_pred = tf.nn.sigmoid(y_pred_logits).numpy()
    
    print(f"预测完成!")
    print(f"  形状: y_true={y_true.shape}, y_pred={y_pred.shape}")
    print(f"  总样本数: {y_true.shape[0]} (应该约为 {(625+6000)*10} = 66250)")
    
    endpoints = data_config.month_endpoints
    print(f"  时间端点: {endpoints} 个月")
    
    all_results = {}
    
    for i, endpoint in enumerate(endpoints):
        print(f"\n=== 分析时间端点: {endpoint}个月 (索引 {i}) ===")
        
        y_true_endpoint = y_true[:, i]
        y_pred_endpoint = y_pred[:, i]
        
        print(f"数据统计:")
        print(f"  总样本数: {len(y_true_endpoint)}")
        print(f"  正样本数: {np.sum(y_true_endpoint)} ({100*np.mean(y_true_endpoint):.2f}%)")
        print(f"  预测概率范围: {y_pred_endpoint.min():.4f} - {y_pred_endpoint.max():.4f}")
        
        print(f"开始{endpoint}个月端点的阈值分析...")
        results_df = analyze_precision_recall_at_thresholds(
            y_true_endpoint, y_pred_endpoint, 
            start_threshold=0.001,
            end_threshold=0.999,
            step=0.001
        )
        
        results_df['endpoint_months'] = endpoint
        all_results[endpoint] = results_df
        
        output_file = f"threshold_analysis_{endpoint}months.csv"
        results_df.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")
        
        print(f"\n=== {endpoint}个月端点分析结果 ===")
        
        best_f1_idx = results_df['f1_score'].idxmax()
        best_f1_result = results_df.loc[best_f1_idx]
        
        print(f"最佳F1 Score: {best_f1_result['f1_score']:.4f}")
        print(f"  阈值: {best_f1_result['threshold']:.4f}")
        print(f"  Precision: {best_f1_result['precision']:.4f}")
        print(f"  Recall: {best_f1_result['recall']:.4f}")
        
        print(f"\n关键阈值结果:")
        sample_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        for thresh in sample_thresholds:
            if thresh <= results_df['threshold'].max():
                idx = results_df['threshold'].sub(thresh).abs().idxmin()
                row = results_df.loc[idx]
                print(f"  阈值 {row['threshold']:.3f}: Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1_score']:.4f}")
    
    combined_results = pd.concat(all_results.values(), ignore_index=True)
    combined_output = "threshold_analysis_all_endpoints.csv"
    combined_results.to_csv(combined_output, index=False)
    
    print(f"\n=== 总结 ===")
    print(f"已完成所有 {len(endpoints)} 个时间端点的阈值分析")
    print(f"各端点结果文件: threshold_analysis_{{endpoint}}months.csv")
    print(f"合并结果文件: {combined_output}")
    print("可以在Excel中打开查看详细的阈值分析结果")

if __name__ == "__main__":
    main()