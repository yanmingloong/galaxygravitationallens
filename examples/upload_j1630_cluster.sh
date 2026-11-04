#!/bin/bash
# J1630 椭圆透镜上集群：更新 galaxylens 代码（椭圆支持）+ 上传 HST 数据 + slurm + 提交。
# 用法：bash examples/upload_j1630_cluster.sh
# 注意：给用户的命令用花括号 {..} 展开（单行），别用 * 通配符（WSL 粘贴会吞）。

CLUSTER="<user>@<cluster-host>"  # edit
REMOTE="/home/<user>/galaxylens"                               # 集群 galaxylens 仓库
REMOTE_DATA="/path/to/cluster/data/J1630+4520"                 # 数据盘：HST FITS

ROOT="$(cd "$(dirname "$0")/.." && pwd)"                        # galaxylens 仓库根
DATA_LOCAL="/path/to/local/data/J1630+4520"                    # 本地 HST FITS

set -e

# 1. 建目录（galaxylens 仓库根 + 数据盘）
ssh "$CLUSTER" "mkdir -p $REMOTE_DATA $REMOTE/configs/fit $REMOTE/examples $REMOTE/src"

# 2. 整目录同步 src/galaxylens（保证集群代码与本地完全一致，不漏任何子模块）
scp -r "$ROOT/src/galaxylens" "$CLUSTER:$REMOTE/src/"

# 3. 上传 J1630 配方
scp "$ROOT/configs/fit/j1630_ellip.yaml" \
    "$CLUSTER:$REMOTE/configs/fit/"

# 4. 上传 HST F814W 数据（sci/noise/psf/starmask 四件套）
scp "$DATA_LOCAL/1630+4520_f814w_sci.fits" \
    "$DATA_LOCAL/1630+4520_f814w_noise.fits" \
    "$DATA_LOCAL/1630+4520_f814w_psf.fits" \
    "$DATA_LOCAL/1630+4520_f814w_starmask.fits" \
    "$CLUSTER:$REMOTE_DATA/"

# 5. 上传 slurm
scp "$ROOT/examples/j1630_ellip.slurm" \
    "$CLUSTER:$REMOTE/examples/"

echo "=== 上传完成，提交作业 ==="
ssh "$CLUSTER" "cd $REMOTE && export PATH=/usr/local/slurm/bin:\$PATH && sbatch examples/j1630_ellip.slurm"
