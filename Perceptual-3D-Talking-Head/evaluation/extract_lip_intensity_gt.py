import argparse
import os
import re
import numpy as np


# 口唇区域顶点索引
mouth_map = [
    1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1590, 1591, 1593, 1593,
    1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1686, 1687, 1691, 1693,
    1694, 1695, 1696, 1697, 1700, 1702, 1703, 1704, 1709, 1710, 1711, 1712, 1713,
    1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1728, 1729, 1730,
    1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750,
    1751, 1758, 1763, 1765, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779,
    1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1801,
    1802, 1803, 1804, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1865, 1866,
    2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2726, 2726, 2727, 2729, 2729,
    2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810,
    2811, 2812, 2813, 2814, 2817, 2819, 2820, 2821, 2826, 2827, 2828, 2829, 2830,
    2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2843, 2844, 2845,
    2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865,
    2866, 2869, 2871, 2873, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886,
    2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2904,
    2905, 2906, 2907, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2948, 2949,
    3503, 3504, 3506, 3509, 3511, 3512, 3513, 3531, 3533, 3537, 3541, 3543, 3546,
    3547, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801,
    3802, 3803, 3804, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921,
    3922, 3923, 3924, 3925, 3926, 3927, 3928
]


EMOTION_MAP = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "surprised",
    "4": "fear",
    "5": "disgusted",
    "6": "angry",
    "7": "contempt",
}

INTENSITY_MAP = {
    "0": "level_1",
    "1": "level_2",
    "2": "level_3",
}


def calculate_facial_movement_intensity(vertice_path: str) -> float:
    """计算单条顶点序列的口唇运动强度（RMS 位移）。"""
    keypoints = np.load(vertice_path, allow_pickle=True)
    keypoints = np.array(keypoints).reshape(-1, 5023, 3)
    keypoints = keypoints[:, mouth_map, :]

    if keypoints.shape[0] < 2:
        return 0.0

    displacements = keypoints[1:] - keypoints[:-1]
    displacement_magnitudes = np.sqrt(np.sum(displacements ** 2, axis=2))
    movement_intensity_per_frame = np.mean(displacement_magnitudes, axis=1)
    return float(np.sqrt(np.mean(movement_intensity_per_frame ** 2)))


def parse_raw_mead_name(filename: str):
    """解析原始顶点命名：{id}_{clip}_{emotion_id}_{intensity_id}.npy"""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) != 4:
        return None

    identity, clip, emotion_id, intensity_id = parts
    if not re.fullmatch(r"[MW]\d{3}", identity):
        return None
    if not clip.isdigit():
        return None
    if emotion_id not in EMOTION_MAP or intensity_id not in INTENSITY_MAP:
        return None

    return {
        "identity": identity,
        "clip": clip,
        "emotion": EMOTION_MAP[emotion_id],
        "level": INTENSITY_MAP[intensity_id],
    }


def main():
    parser = argparse.ArgumentParser(description="从原始顶点提取口唇强度并生成 SLCC 可用目录")
    parser.add_argument(
        "--source-root",
        default=os.path.expanduser("~/datasets/mead_clean/vertex"),
        help="原始顶点 .npy 根目录",
    )
    parser.add_argument(
        "--destination-root",
        default="./lip_disp_gt",
        help="输出目录（结构将符合 evaluate_SLCC.py）",
    )
    parser.add_argument(
        "--csv-name",
        default="gt.csv",
        help="每个 clip 下输出的 csv 文件名",
    )
    args = parser.parse_args()

    processed = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(args.source_root):
        for filename in filenames:
            if not filename.endswith(".npy"):
                continue

            meta = parse_raw_mead_name(filename)
            if meta is None:
                skipped += 1
                print(f"[跳过] 文件名不符合原始顶点规范: {filename}")
                continue

            src_path = os.path.join(dirpath, filename)
            dst_dir = os.path.join(
                args.destination_root,
                meta["identity"],
                meta["emotion"],
                meta["level"],
                meta["clip"],
            )
            dst_path = os.path.join(dst_dir, args.csv_name)

            os.makedirs(dst_dir, exist_ok=True)
            intensity = calculate_facial_movement_intensity(src_path)
            np.savetxt(dst_path, [intensity], delimiter=",")

            processed += 1
            print(f"[完成] {src_path} -> {dst_path}")

    print("\n=== 统计 ===")
    print(f"处理完成: {processed}")
    print(f"跳过数量: {skipped}")
    print(f"输出目录: {os.path.abspath(args.destination_root)}")


if __name__ == "__main__":
    main()
