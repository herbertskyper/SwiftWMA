#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
import PIL.Image

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def process_video_psnr(gt_path, pred_path, resize=None):
    try:
        clip_gt = VideoFileClip(gt_path)
        clip_pred = VideoFileClip(pred_path)

        fps = min(clip_gt.fps, clip_pred.fps)
        duration = min(clip_gt.duration, clip_pred.duration)
        
        # Calculate number of frames based on duration and fps
        time_points = np.arange(0, duration, 1.0 / fps)

        video_psnrs = []
        frame_indices = []

        print(f"Processing {len(time_points)} frames...")
        
        for i, t in enumerate(time_points):
            frame_gt = clip_gt.get_frame(t)
            frame_pred = clip_pred.get_frame(t)

            if resize:
                img_gt = PIL.Image.fromarray(frame_gt).resize(resize, PIL.Image.Resampling.BILINEAR)
                img_pred = PIL.Image.fromarray(frame_pred).resize(resize, PIL.Image.Resampling.BILINEAR)
                psnr = calculate_psnr(np.array(img_gt), np.array(img_pred))
            else:
                 # Ensure shapes match if not resizing to fixed dim
                if frame_gt.shape != frame_pred.shape:
                    # Resize pred to match gt
                    img_gt = PIL.Image.fromarray(frame_gt)
                    img_pred = PIL.Image.fromarray(frame_pred).resize(img_gt.size, PIL.Image.Resampling.BILINEAR)
                    psnr = calculate_psnr(np.array(img_gt), np.array(img_pred))
                else:
                    psnr = calculate_psnr(frame_gt, frame_pred)
            
            video_psnrs.append(psnr)
            frame_indices.append(i)

        clip_gt.close()
        clip_pred.close()

        return frame_indices, video_psnrs

    except Exception as e:
        print(f"Error processing {os.path.basename(gt_path)}: {e}")
        return None, None

def plot_psnr(frame_indices, psnrs, output_path, title):
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, psnrs, 'bo-', label='Pre-encode PSNR') # 'bo-' for blue circles and lines, label from example
    plt.title(f"Frame-wise PSNR: {title}")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate PSNR plot from GT and Pred videos")
    parser.add_argument('--gt_video', type=str, required=True, help='path to reference video')
    parser.add_argument('--pred_video', type=str, required=True, help='path to predicted video')
    parser.add_argument('--output_plot', type=str, default='psnr_plot.png', help='path to output plot image')
    parser.add_argument('--resize', action='store_true', help='Resize frames to 256x256 (as in challenge script)')
    args = parser.parse_args()

    if not os.path.exists(args.gt_video):
        print(f"Error: GT video not found at {args.gt_video}")
        return
    if not os.path.exists(args.pred_video):
        print(f"Error: Pred video not found at {args.pred_video}")
        return

    # Default to 256x256 only if requested, otherwise full res or matching res
    resize_dim = (256, 256) if args.resize else None
    
    print(f"Comparing:\nRef: {args.gt_video}\nPred: {args.pred_video}")
    frame_indices, psnrs = process_video_psnr(args.gt_video, args.pred_video, resize=resize_dim)

    if psnrs:
        title = os.path.basename(args.gt_video)
        plot_psnr(frame_indices, psnrs, args.output_plot, title)
        print(f"Average PSNR: {np.mean(psnrs):.4f} dB")
    else:
        print("Failed to calculate PSNRs.")

if __name__ == "__main__":
    main()
