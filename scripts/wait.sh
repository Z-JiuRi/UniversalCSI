#!/bin/bash

while true; do
    # 统计匹配的进程数（使用 [p] 技巧避免 grep 自身被计入）
    count=$(ps -u | grep "[p]ython ./main.py --exp_name seed42/COST2100/in" | wc -l)
    
    if [ "$count" -eq 0 ]; then
        echo "$(date): No matching processes found. Running scripts/run.sh ..."
        bash scripts/run.sh
        break
    else
        echo "$(date): Found $count process(es) still running. Waiting 5 minutes..."
        sleep 300   # 300 秒 = 5 分钟
    fi
done