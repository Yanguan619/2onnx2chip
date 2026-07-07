#!/bin/bash
# Usage: ./monitor-mem.sh [interval] [count]
# 监控已用内存 (系统 + NPU0)，带增量变化（基于首次采样）
command -v column >/dev/null 2>&1 || apt update && apt-get install -y bsdmainutils

INTERVAL=${1:-0.5}
COUNT=${2:-0}

i=0
FIRST_USED=""
FIRST_NPU0=""

# 创建临时文件存储所有行
TMPFILE=$(mktemp)

# 写入表头
echo -e "Time\tUsedMem(GB)\tMemDelta(GB)\tNPU0(GB)\tNPU0_Delta(GB)\tTotalDelta(GB)" >> "$TMPFILE"

while [ $COUNT -eq 0 ] || [ $i -lt $COUNT ]; do
    TS=$(date +%H:%M:%S)

    # 直接从 free -h 取 used 列（第3列），并去除单位后缀
    USED_RAW=$(free -h | awk '/^Mem:/ {print $3}')
    USED_GB=$(echo "$USED_RAW" | sed 's/[^0-9.]//g')

    # 如果为空或非数字，则跳过本次采样
    if ! echo "$USED_GB" | grep -qE '^[0-9]+\.?[0-9]*$'; then
        USED_GB="0.00"
    fi

    # 如果是第一次采样，记录基准值
    if [ -z "$FIRST_USED" ]; then
        FIRST_USED="$USED_GB"
        MEM_DELTA="0.00"
    else
        MEM_DELTA=$(awk "BEGIN{printf \"%.2f\", $USED_GB - $FIRST_USED}")
    fi

    # NPU0 内存采集
    NPU0_LINE=$(npu-smi info -t memory -o csv 2>/dev/null | awk 'NR==2')
    USED_NPU=$(echo "$NPU0_LINE" | awk -F, '{gsub(/[^0-9.]/,"",$3); print $3}')
    if echo "$USED_NPU" | grep -qE '^[0-9]+\.?[0-9]*$' 2>/dev/null; then
        USED_GB_NPU=$(awk "BEGIN{printf \"%.2f\", $USED_NPU / 1024}")

        # 如果是第一次采样，记录基准值
        if [ -z "$FIRST_NPU0" ]; then
            FIRST_NPU0="$USED_GB_NPU"
            DELTA_NPU="0.00"
        else
            DELTA_NPU=$(awk "BEGIN{printf \"%.2f\", $USED_GB_NPU - $FIRST_NPU0}")
        fi
    else
        USED_GB_NPU="N/A"
        DELTA_NPU="0.00"
    fi

    # 计算总增量（只有两者都是有效数字时才求和）
    if [ "$USED_GB_NPU" != "N/A" ] && [ -n "$FIRST_USED" ] && [ -n "$FIRST_NPU0" ]; then
        TOTAL_DELTA=$(awk "BEGIN{printf \"%.2f\", $MEM_DELTA + $DELTA_NPU}")
    else
        TOTAL_DELTA="0.00"
    fi

    # 写入数据行到临时文件
    echo -e "$TS\t$USED_GB\t$MEM_DELTA\t${USED_GB_NPU}\t$DELTA_NPU\t$TOTAL_DELTA" >> "$TMPFILE"

    # 清屏并重新用 column 显示整个文件
    clear
    column -t -s $'\t' "$TMPFILE"

    i=$((i+1))
    [ $COUNT -eq 0 ] || [ $i -lt $COUNT ] && sleep "$INTERVAL"
done

# 清理临时文件
rm -f "$TMPFILE"
