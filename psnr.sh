# !/bin/bash
conda activate /home/skyper/.conda/envs/unifolm-wma
sample=unitree_z1_dual_arm_stackbox_v2
caseid=1
gt_video_name=${sample}_case${caseid}.mp4
gt_video=./${sample}/case${caseid}/${gt_video_name}
pred_video=./${sample}/case${caseid}/output/inference/5_full_fs4.mp4
python3 ./scripts/generate_psnr_plot.py --gt_video ${gt_video} --pred_video ${pred_video}