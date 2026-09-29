"""PSF stage 修复的单元测试 (2026-08-17)。

覆盖:
  1. psf_ee50 离散半光半径 (窄高斯 < 宽高斯, 排序正确)
  2. filter_field_stars 孤立性筛选 (KD-tree)
  3. filter_field_stars 贴边排除

用法:
  cd GalaxyGravLens && PYTHONPATH="DataPipeline:$PYTHONPATH" \
    python DataPipeline/pipeline/tests/test_psf_selection.py
"""
import numpy as np

from galaxylens.data_pipeline.imaging.stages.psf import psf_ee50, psf_ellipticity, filter_field_stars


def test_psf_ee50_ordering():
    """窄高斯 EE50 < 宽高斯 EE50 (相对排序正确, 用于选优门槛).

    注意: psf_ee50 是离散半光半径 (像素中心距离排序), 比连续解析值
    略小, 但跨 PSF 一致, 相对比较和门槛判定不受影响。
    """
    yy, xx = np.mgrid[-15:16, -15:16]
    narrow = np.exp(-(xx**2 + yy**2) / (2 * (2.5 / 2.355)**2))  # FWHM=2.5px
    broad = np.exp(-(xx**2 + yy**2) / (2 * (5.0 / 2.355)**2))   # FWHM=5.0px
    narrow /= narrow.sum()
    broad /= broad.sum()
    ee_n, fw_n = psf_ee50(narrow)
    ee_b, fw_b = psf_ee50(broad)
    assert ee_n < ee_b, f"窄高斯 EE50={ee_n:.2f} 应 < 宽高斯 {ee_b:.2f}"
    assert 0.5 < ee_n < 2.0, f"窄高斯 EE50={ee_n:.2f} 异常"
    print(f"  ✅ psf_ee50: 窄={ee_n:.2f}px < 宽={ee_b:.2f}px (排序正确)")


def test_filter_isolation():
    """2 颗拥挤星(间距 2px)应被剔除, 1 颗孤立星保留."""
    stars = [(100, 100, 50.0), (102, 100, 40.0), (200, 200, 30.0)]
    sel = filter_field_stars(stars, center=(300, 300),
                             exclusion_radius_pix=10, data_shape=(300, 300),
                             min_dist=5.0)
    assert len(sel) == 1 and sel[0][2] == 30.0, f"孤立性筛选错误: {sel}"
    print(f"  ✅ 孤立性: 3颗(2拥挤) -> 1颗保留")


def test_filter_edge():
    """贴边源 (y=5 < edge_margin=20) 应被排除."""
    sel = filter_field_stars([(150, 5, 99.0)], center=(300, 300),
                             exclusion_radius_pix=10, data_shape=(300, 300),
                             edge_margin=20.0)
    assert len(sel) == 0, f"贴边排除错误: {sel}"
    print("  ✅ 贴边排除: y=5 星被剔除")


def test_ellipticity_circular():
    """圆高斯椭率应 ≈ 0."""
    yy, xx = np.mgrid[-10:11, -10:11]
    g = np.exp(-(xx**2 + yy**2) / (2 * 1.0**2))
    g /= g.sum()
    e = psf_ellipticity(g)
    assert e < 0.02, f"圆高斯椭率={e:.4f} 应≈0"
    print(f"  ✅ 椭率: 圆高斯 e={e:.4f}")


if __name__ == "__main__":
    test_psf_ee50_ordering()
    test_filter_isolation()
    test_filter_edge()
    test_ellipticity_circular()
    print("\n✅ 全部 4 项单元测试通过")
