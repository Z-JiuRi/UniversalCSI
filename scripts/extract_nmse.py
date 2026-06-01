#!/usr/bin/env python3
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

def find_run_logs(root_dir="."):
    """递归查找所有 run.log 文件"""
    for path in Path(root_dir).rglob("run.log"):
        yield path

def parse_run_log(log_path):
    """解析单个 run.log 文件，返回 (encoder, decoder, best_nmse) 或 None"""
    encoder = decoder = None
    nmse_values = []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 提取 encoder/decoder 配置行（示例：encoder=mlp_mixer; decoder=hybrid;）
            config_match = re.search(r'encoder=([^;]+);\s*decoder=([^;]+);', content)
            if config_match:
                encoder = config_match.group(1).strip()
                decoder = config_match.group(2).strip()
            else:
                return None  # 没有配置信息，跳过
            
            # 提取所有 Best NMSE 值
            # 匹配模式：Best NMSE: 3.1883e+01 (epoch=10)
            pattern = r'Best NMSE:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            for match in re.finditer(pattern, content):
                try:
                    val = float(match.group(1))
                    nmse_values.append(val)
                except ValueError:
                    continue
            
            if nmse_values:
                best_nmse = min(nmse_values)
                return (encoder, decoder, best_nmse)
    except Exception as e:
        print(f"Error reading {log_path}: {e}", file=sys.stderr)
    return None

def build_matrix(data):
    """data: list of (encoder, decoder, nmse) 构建 DataFrame"""
    df = pd.pivot_table(
        pd.DataFrame(data, columns=['encoder', 'decoder', 'nmse']),
        index='decoder',
        columns='encoder',
        values='nmse',
        aggfunc='min'  # 同一对 (encoder, decoder) 可能有多个值，取最小
    )
    return df

def main(root_dir="."):
    results = []
    for log_path in find_run_logs(root_dir):
        parsed = parse_run_log(log_path)
        if parsed:
            encoder, decoder, nmse = parsed
            results.append((encoder, decoder, nmse))
    
    if not results:
        print("No valid data found.")
        return
    
    df = build_matrix(results)
    
    # 输出到控制台（表格形式）
    print("NMSE Matrix (rows=decoder, columns=encoder):")
    print(df.to_string(float_format="%.4e", na_rep="--"))
    
    # 保存为 CSV
    csv_file = "nmse_matrix.csv"
    df.to_csv(csv_file, float_format="%.6e", na_rep="")
    print(f"\nMatrix saved to {csv_file}")

if __name__ == "__main__":
    # 可以指定根目录，默认为当前目录
    root = sys.argv[1] if len(sys.argv) > 1 else "/storage/hujiacong/zxd/Huawei/UniversalCSI/exps/seed42/WAIRD"
    main(root)