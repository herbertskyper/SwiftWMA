#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_real_seconds(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # 匹配形如: real    2m29.981s
    matches = re.findall(r"^real\s+(\d+)m([\d.]+)s\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"未在 {log_path} 中找到 real 时间")
    m, s = matches[-1]
    return float(m) * 60.0 + float(s)


def find_pred_video(case_result_dir: Path) -> Path:
    # 只取 case 目录顶层 mp4（与你 gather 逻辑一致）
    mp4s = sorted(case_result_dir.glob("*.mp4"))
    if len(mp4s) != 1:
        raise ValueError(f"{case_result_dir} 顶层 mp4 数量应为 1，实际为 {len(mp4s)}")
    return mp4s[0]


def calc_psnr(repo_root: Path, gt_video: Path, pred_video: Path) -> float:
    psnr_script = repo_root / "psnr_score_for_challenge.py"
    if not psnr_script.exists():
        raise FileNotFoundError(f"找不到 {psnr_script}")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tf:
        cmd = [
            sys.executable, str(psnr_script),
            "--gt_video", str(gt_video),
            "--pred_video", str(pred_video),
            "--output_file", tf.name
        ]
        subprocess.run(cmd, check=True, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tf.seek(0)
        data = json.load(tf)
    return float(data["psnr"])


def main():
    parser = argparse.ArgumentParser(description="检查 results 与 summary.json 一致性")
    parser.add_argument("--repo-root", type=str, default=".", help="仓库根目录")
    parser.add_argument("--summary", type=str, default="results/summary.json", help="summary.json 路径")
    parser.add_argument("--time-tol", type=float, default=1e-3, help="时间容差(秒)")
    parser.add_argument("--psnr-tol", type=float, default=1e-3, help="PSNR容差")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    summary_path = (repo_root / args.summary).resolve()

    if not summary_path.exists():
        print(f"[ERROR] summary 文件不存在: {summary_path}")
        sys.exit(2)

    records = json.loads(summary_path.read_text(encoding="utf-8"))
    total = len(records)
    failed = 0

    for rec in records:
        scenario = rec["scenario"]
        case_id = int(rec["case_id"])
        case_name = f"case{case_id}"

        expected_time = float(rec["time_after_optim"])
        expected_psnr = float(rec["psnr"])

        case_result_dir = repo_root / "results" / scenario / case_name
        log_file = case_result_dir / "output.log"
        gt_video = repo_root / scenario / case_name / f"{scenario}_{case_name}.mp4"

        print(f"\n[CHECK] {scenario}/{case_name}")

        try:
            if not case_result_dir.exists():
                raise FileNotFoundError(f"目录不存在: {case_result_dir}")
            if not log_file.exists():
                raise FileNotFoundError(f"文件不存在: {log_file}")
            if not gt_video.exists():
                raise FileNotFoundError(f"GT视频不存在: {gt_video}")

            pred_video = find_pred_video(case_result_dir)
            real_sec = parse_real_seconds(log_file)
            psnr_val = calc_psnr(repo_root, gt_video, pred_video)

            time_ok = abs(real_sec - expected_time) <= args.time_tol
            psnr_ok = abs(psnr_val - expected_psnr) <= args.psnr_tol

            print(f"  real(sec): actual={real_sec:.3f}, summary={expected_time:.3f}, ok={time_ok}")
            print(f"  psnr     : actual={psnr_val:.4f}, summary={expected_psnr:.4f}, ok={psnr_ok}")

            if not (time_ok and psnr_ok):
                failed += 1

        except Exception as e:
            failed += 1
            print(f"  [FAIL] {e}")

    print(f"\n====== 检查完成: total={total}, failed={failed}, passed={total - failed} ======")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()