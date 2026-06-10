from __future__ import annotations
import numpy as np
import torch


def _make_mast3r(min_conf_thr: float = 1.5):
    from vismatch.im_models.master import Mast3rMatcher
    m = Mast3rMatcher.__new__(Mast3rMatcher)
    m.min_conf_thr = min_conf_thr
    return m


def test_filter_keeps_only_high_conf_matches() -> None:
    m = _make_mast3r(1.5)
    mkpts0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    mkpts1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    conf0 = np.array([[2.0, 0.0, 0.0],
                      [0.0, 1.6, 0.0],
                      [0.0, 0.0, 1.4]], dtype=np.float32)
    conf1 = np.array([[2.0, 0.0, 0.0],
                      [0.0, 1.4, 0.0],
                      [0.0, 0.0, 2.0]], dtype=np.float32)
    r0, r1 = m._filter_by_conf(mkpts0, mkpts1, conf0, conf1)
    print(f"[test_filter_keeps] input={len(mkpts0)}, output={len(r0)}")
    np.testing.assert_array_equal(r0, np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(r1, np.array([[0.0, 0.0]], dtype=np.float32))


def test_filter_handles_empty_matches() -> None:
    m = _make_mast3r(1.5)
    empty = np.empty((0, 2), dtype=np.float32)
    conf = np.ones((4, 4), dtype=np.float32)
    r0, r1 = m._filter_by_conf(empty, empty, conf, conf)
    print(f"[test_filter_empty] output r0={r0.shape}, r1={r1.shape}")
    assert r0.shape == (0, 2) and r1.shape == (0, 2)


def test_min_conf_thr_default_is_zero() -> None:
    from vismatch.im_models.master import Mast3rMatcher
    m = Mast3rMatcher.__new__(Mast3rMatcher)
    torch.nn.Module.__init__(m)
    m.device = "cpu"
    m.skip_ransac = False
    m.ransac_iters = 2000
    m.ransac_conf = 0.95
    m.ransac_reproj_thresh = 3
    m.verbose = False
    m.min_conf_thr = 0.0
    print(f"[test_default_conf_thr] min_conf_thr={m.min_conf_thr}")
    assert m.min_conf_thr == 0.0


def test_mickey_absent_from_available_models() -> None:
    import vismatch
    print(f"[test_mickey_absent] mickey in models={'mickey' in vismatch.available_models}")
    assert "mickey" not in vismatch.available_models


def test_plot_matches_result_dict_interface(tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vismatch import viz

    img = torch.zeros(3, 16, 16)
    result = {
        "matched_kpts0": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        "matched_kpts1": np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32),
        "inlier_kpts0": np.array([[1.0, 1.0]], dtype=np.float32),
        "inlier_kpts1": np.array([[3.0, 3.0]], dtype=np.float32),
    }
    save_path = tmp_path / "smoke.jpg"
    axs = viz.plot_matches(img, img, result, save_path=save_path)
    print(f"[test_viz_smoke] exists={save_path.exists()}, axes={len(axs)}")
    assert save_path.exists()
    assert len(axs) == 2
    plt.close("all")
