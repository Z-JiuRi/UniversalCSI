#!/bin/bash
# 暂停/恢复所有 GPU 上命令行包含 "lr1e-4" 的进程
# 用法:
#   bash gpu_lr1e4_control.sh pause    # 暂停 (SIGSTOP)
#   bash gpu_lr1e4_control.sh resume   # 恢复 (SIGCONT)
#   bash gpu_lr1e4_control.sh list     # 只列出匹配进程，不操作

set -euo pipefail

MATCH_PATTERN="lr1e-4"

usage() {
    echo "用法: $0 {pause|resume|list}"
    echo ""
    echo "  pause   暂停所有匹配进程 (SIGSTOP)"
    echo "  resume  恢复所有匹配进程 (SIGCONT)"
    echo "  list    列出匹配进程但不操作"
    exit 1
}

[[ $# -ne 1 ]] && usage

ACTION="$1"
[[ "$ACTION" != "pause" && "$ACTION" != "resume" && "$ACTION" != "list" ]] && usage

# ---------- 查找匹配进程 ----------
# 排除当前脚本自身和 grep 进程
PIDS=$(ps aux | grep -i "$MATCH_PATTERN" | grep -v grep | grep -v "$(basename $0)" | awk '{print $2}' | sort -u)

if [[ -z "$PIDS" ]]; then
    echo "[INFO] 未找到包含 \"$MATCH_PATTERN\" 的进程"
    exit 0
fi

# ---------- 收集进程详细信息 ----------
echo "=========================================="
echo "匹配 \"$MATCH_PATTERN\" 的进程:"
echo "------------------------------------------"
printf "%-8s %-10s %s\n" "PID" "GPU(s)" "CMD (截断)"
echo "------------------------------------------"

declare -a PID_LIST=()
while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    PID_LIST+=("$pid")

    # 获取命令行 (截断显示)
    if [[ -r /proc/$pid/cmdline ]]; then
        cmd=$(tr '\0' ' ' < /proc/$pid/cmdline | head -c 120)
    else
        cmd="(不可读)"
    fi

    # 通过 nvidia-smi 查询该进程在哪些 GPU 上
    gpu_info=""
    if command -v nvidia-smi &>/dev/null; then
        gpu_info=$(nvidia-smi --query-compute-apps=pid,gpu_name,gpu_bus_id --format=csv,noheader 2>/dev/null | \
            awk -v pid="$pid" -F',' '$1+0==pid { gsub(/^[ \t]+/,"",$2); printf "%s ",$2 }')
    fi
    [[ -z "$gpu_info" ]] && gpu_info="(未检测到)"

    printf "%-8s %-10s %s\n" "$pid" "$gpu_info" "$cmd"
done <<< "$PIDS"

TOTAL=${#PID_LIST[@]}
echo "------------------------------------------"
echo "总计: $TOTAL 个进程"
echo "=========================================="

# ---------- 执行操作 ----------
if [[ "$ACTION" == "list" ]]; then
    exit 0
fi

SIGNAL=""
SIGNAME=""
if [[ "$ACTION" == "pause" ]]; then
    SIGNAL="SIGSTOP"
    SIGNAME="暂停"
elif [[ "$ACTION" == "resume" ]]; then
    SIGNAL="SIGCONT"
    SIGNAME="恢复"
fi

echo ""
echo "[操作] 将 $SIGNAME $TOTAL 个进程 (发送 $SIGNAL)..."
for pid in "${PID_LIST[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -"$SIGNAL" "$pid"
        echo "  [OK] PID $pid 已$SIGNAME"
    else
        echo "  [SKIP] PID $pid 已不存在"
    fi
done

echo "[完成]"
