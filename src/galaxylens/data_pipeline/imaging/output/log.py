"""
处理日志 Processing log
========
记录处理过程的完整元数据到 JSON 文件。

功能:
  1. 创建结构化处理日志
  2. 记录所有参数、方法和输入/输出路径
  3. 支持追加波段信息

使用示例:
  >>> from galaxylens.data_pipeline.imaging.output.log import ProcessingLog
  >>> log = ProcessingLog("J2141-0001", "v2_flc_whtonly")
  >>> log.add_band("F814W", band_info)
  >>> log.save(output_dir)
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ProcessingLog:
    """单次处理运行的完整日志。"""

    def __init__(
        self,
        lens_id: str,
        version_tag: str,
    ) -> None:
        """初始化处理日志。

        Args:
            lens_id: 透镜标识符
            version_tag: 版本标签 (如 "v2_flc_whtonly")
        """
        self.lens_id = lens_id
        self.version_tag = version_tag
        self.start_time = datetime.now()
        self.bands: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []

    def add_band(self, band: str, info: dict[str, Any]) -> None:
        """记录一个波段的处理信息。

        Args:
            band: 波段名
            info: 包含以下键的字典:
              - source_files: 源文件列表
              - product_type: 数据产品类型
              - n_exposures: 曝光数
              - pixel_scale: 像元尺度
              - cutout_shape: 裁剪后 shape
              - centroid: [cx, cy]
              - bg_median: 背景中值
              - bg_rms: 背景 RMS
              - noise_method: 噪声方法
              - psf_source: PSF 来源
              - psf_shape: PSF shape
              - starmask_pixels: mask 像素数
              - cr_fraction_pct: CR 占比
              - quality: 质量评估 (可选)
        """
        self.bands[band] = {
            **info,
            "processed_at": datetime.now().isoformat(),
        }

    def add_error(self, band: str, error: str) -> None:
        """记录处理错误。

        Args:
            band: 波段名
            error: 错误描述
        """
        self.errors.append(f"[{band}] {error}")

    def to_dict(self) -> dict[str, Any]:
        """将日志导出为字典。"""
        return {
            "lens": self.lens_id,
            "version": self.version_tag,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "bands": self.bands,
            "errors": self.errors if self.errors else None,
        }

    def save(self, output_dir: Path) -> Path:
        """保存日志到 JSON 文件。

        Args:
            output_dir: 输出目录

        Returns:
            日志文件路径
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return log_path
