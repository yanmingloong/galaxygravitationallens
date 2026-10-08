"""
Processing log
========
Record the complete metadata of the processing run to a JSON file.

Features:
  1. Create a structured processing log
  2. Record all parameters, methods, and input/output paths
  3. Support appending band information

Usage example:
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
    """Complete log for a single processing run."""

    def __init__(
        self,
        lens_id: str,
        version_tag: str,
    ) -> None:
        """Initialize the processing log.

        Args:
            lens_id: Lens identifier
            version_tag: Version tag (e.g. "v2_flc_whtonly")
        """
        self.lens_id = lens_id
        self.version_tag = version_tag
        self.start_time = datetime.now()
        self.bands: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []

    def add_band(self, band: str, info: dict[str, Any]) -> None:
        """Record the processing information for one band.

        Args:
            band: Band name
            info: Dictionary containing the following keys:
              - source_files: List of source files
              - product_type: Data product type
              - n_exposures: Number of exposures
              - pixel_scale: Pixel scale
              - cutout_shape: Cutout shape
              - centroid: [cx, cy]
              - bg_median: Background median
              - bg_rms: Background RMS
              - noise_method: Noise method
              - psf_source: PSF source
              - psf_shape: PSF shape
              - starmask_pixels: Number of masked pixels
              - cr_fraction_pct: CR fraction
              - quality: Quality assessment (optional)
        """
        self.bands[band] = {
            **info,
            "processed_at": datetime.now().isoformat(),
        }

    def add_error(self, band: str, error: str) -> None:
        """Record a processing error.

        Args:
            band: Band name
            error: Error description
        """
        self.errors.append(f"[{band}] {error}")

    def to_dict(self) -> dict[str, Any]:
        """Export the log as a dictionary."""
        return {
            "lens": self.lens_id,
            "version": self.version_tag,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "bands": self.bands,
            "errors": self.errors if self.errors else None,
        }

    def save(self, output_dir: Path) -> Path:
        """Save the log to a JSON file.

        Args:
            output_dir: Output directory

        Returns:
            Log file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return log_path
