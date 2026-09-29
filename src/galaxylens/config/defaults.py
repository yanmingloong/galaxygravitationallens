"""处理参数默认值（全局常量）。 Processing parameter defaults (global constants)."""
from __future__ import annotations

from typing import Dict

# 裁剪与掩膜
CUTOUT_SIZE_ARCSEC: float = 15.0       # 裁剪视场 (角秒)
MASK_RADIUS_FACTOR: float = 3.0        # 恒星 mask 半径 = N × PSF_FWHM

# PSF 构建
PSF_STAMP_SIZE: int = 31               # PSF stamp 尺寸 (奇数像素)
MIN_STAR_SNR: float = 8.0              # 恒星检测 SNR 阈值
STAR_SHARPLO: float = 0.7              # 场星锐度下限
N_STARS_PSF: int = 5                   # 用于 PSF 的恒星数量

# PSF 对比选优门槛 (EE50 正常 ≈ 1.0-1.5px, 膨胀 > 3px 判定污染; 椭率 > 0.15 判定畸形)
PSF_EE50_MIN_PX: float = 0.8
PSF_EE50_MAX_PX: float = 3.0
PSF_ELL_MAX: float = 0.15

# 仪器像元尺度 (arcsec/pix)
INSTRUMENT_PIXEL_SCALE: Dict[str, float] = {
    "ACS/WFC":   0.050,
    "WFPC2/PC":  0.0455,
    "WFC3/IR":   0.128,
}
