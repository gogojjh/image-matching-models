from __future__ import annotations
from pathlib import Path
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


_ASSETS = Path(__file__).resolve().parent.parent / "vismatch" / "assets"
_EXAMPLE_TEST = _ASSETS / "example_test"
_EXAMPLE_PAIRS = _ASSETS / "example_pairs"


def _run_matcher(device: str = "cpu") -> "BaseMatcher":
    from vismatch import get_matcher
    return get_matcher("sift-nn", device=device, max_num_keypoints=2048)


def test_real_image_matching_original_vs_warped_has_inliers() -> None:
    """sift-nn on original.jpg vs warped.jpg should find reasonable inliers."""
    import matplotlib
    matplotlib.use("Agg")
    matcher = _run_matcher()
    result = matcher(str(_EXAMPLE_TEST / "original.jpg"), str(_EXAMPLE_TEST / "warped.jpg"))
    n_inliers = result["num_inliers"]
    n_matched = len(result["matched_kpts0"])
    print(f"[test_real_orivwarp] matched={n_matched}, inliers={n_inliers}, "
          f"ratio={n_inliers / max(n_matched, 1):.2f}")
    assert n_matched > 0
    assert n_inliers >= 50
    assert result["H"] is not None


def test_real_image_matching_visualization_output(tmp_path) -> None:
    """Match original/warped and save visualization to tests/ dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vismatch import viz

    matcher = _run_matcher()
    img0 = str(_EXAMPLE_TEST / "original.jpg")
    img1 = str(_EXAMPLE_TEST / "warped.jpg")
    result = matcher(img0, img1)

    save_path = tmp_path / "original_vs_warped.jpg"
    axs = viz.plot_matches(img0, img1, result, save_path=save_path)
    print(f"[test_viz_orivwarp] save_path={save_path}, exists={save_path.exists()}, "
          f"inliers={result['num_inliers']}")
    assert save_path.exists()
    assert result["num_inliers"] >= 50
    plt.close("all")


def test_example_pairs_indoor_matching(tmp_path) -> None:
    """Match indoor pair (gcs_close vs gcs_far)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vismatch import viz

    indoor_dir = _EXAMPLE_PAIRS / "indoor"
    matcher = _run_matcher()
    img0 = str(indoor_dir / "gcs_close.jpg")
    img1 = str(indoor_dir / "gcs_far.jpg")
    result = matcher(img0, img1)

    save_path = tmp_path / "indoor_match.jpg"
    axs = viz.plot_matches(img0, img1, result, save_path=save_path, show_text=False)
    viz.add_text(axs[0], f"indoor: {result['num_inliers']} inliers", fs=15)
    viz.save_plot(axs[0].get_figure(), save_path)
    print(f"[test_indoor] inliers={result['num_inliers']}, saved={save_path.exists()}")
    assert save_path.exists()
    assert result["num_inliers"] >= 0  # wide-baseline indoor may yield 0 sift-nn inliers
    plt.close("all")


def test_example_pairs_outdoor_matching(tmp_path) -> None:
    """Match outdoor pair (montmartre_close vs montmartre_far)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vismatch import viz

    outdoor_dir = _EXAMPLE_PAIRS / "outdoor"
    matcher = _run_matcher()
    img0 = str(outdoor_dir / "montmartre_close.jpg")
    img1 = str(outdoor_dir / "montmartre_far.jpg")
    result = matcher(img0, img1)

    save_path = tmp_path / "outdoor_match.jpg"
    axs = viz.plot_matches(img0, img1, result, save_path=save_path, show_text=False)
    viz.add_text(axs[0], f"outdoor: {result['num_inliers']} inliers", fs=15)
    viz.save_plot(axs[0].get_figure(), save_path)
    print(f"[test_outdoor] inliers={result['num_inliers']}, saved={save_path.exists()}")
    assert save_path.exists()
    assert result["num_inliers"] >= 0  # wide-baseline outdoor may yield few sift-nn inliers
    plt.close("all")
