# SwiftWMA: Cross-UNet Caching and Efficient Parallelism for World-Model-Action Inference

<p align="center">
  <b>An efficient inference system for autoregressive World-Model-Action models</b>
</p>


## ⚙️ Installation

Create the environment and install the required dependencies:

```bash
conda create -n swiftwma python==3.10.18
conda activate swiftwma

conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

git clone --recurse-submodules https://github.com/herbertskyper/SwiftWMA.git
cd SwiftWMA

git submodule update --init --recursive

pip install -e .

cd external/dlimp
pip install -e .
```

## 🧰 Model Checkpoints

SwiftWMA is built on top of [UnifoLM-WMA-0](https://github.com/unitreerobotics/unifolm-world-model-action) and uses its publicly released model checkpoints.

| Model              | Description                                                                            | Link                                                                     |
| ------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| UnifoLM-WMA-0 Dual | Fine-tuned on Unitree downstream datasets in both decision-making and simulation modes | [HuggingFace](https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual) |

For the experiments in SwiftWMA, please download the corresponding UnifoLM-WMA-0 checkpoint and specify its path in the inference configuration.

## 🛢️ Datasets

We use the following five public Unitree robotic manipulation datasets:

| Dataset                    | Robot      | Link                                                                                                           |
| -------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| Z1_StackBox                | Unitree Z1 | [HuggingFace](https://huggingface.co/datasets/unitreerobotics/Z1_StackBox_Dataset/tree/v2.1)                   |
| Z1_DualArm_StackBox        | Unitree Z1 | [HuggingFace](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset/tree/v2.1)         |
| Z1_DualArm_StackBox_V2     | Unitree Z1 | [HuggingFace](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2/tree/v2.1)      |
| Z1_DualArm_Cleanup_Pencils | Unitree Z1 | [HuggingFace](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_CleanupPencils_Dataset/tree/v2.1)   |
| G1_Pack_Camera             | Unitree G1 | [HuggingFace](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset/tree/v2.1) |

## 📂 Input Data Organization

SwiftWMA follows the input format of UnifoLM-WMA-0.

For interactive world-model inference, organize the input prompts as follows:

```text
world_model_interaction_prompts/
├── images/
│   ├── dataset1_name/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── ...
├── transitions/
│   ├── dataset1_name/
│   │   ├── meta_data/
│   │   ├── 0.h5
│   │   ├── 1.h5
│   │   └── ...
│   └── ...
├── dataset1_name.csv
└── ...
```

The files are used as follows:

* `images/`: image prompts used as visual observations;
* `transitions/`: robot state and action data;
* `meta_data/`: statistics used for normalization;
* `*.csv`: specifies image prompts, language instructions, and the corresponding robot states.

The transition files are mainly used to retrieve the robot state corresponding to each image prompt during interaction inference.

### Using the included test data

The repository includes ready-to-use interaction prompts in:

```text
examples/world_model_interaction_prompts
```

You can use this directory to test the inference pipeline directly without preparing another prompt directory. In `scripts/run_world_model_interaction.sh`, set `prompt_dir` to the repository path:

```bash
prompt_dir="$(pwd)/examples/world_model_interaction_prompts"
```

The included data contains the following datasets, which are already listed in the script's `datasets` array:

```text
unitree_z1_stackbox
unitree_z1_dual_arm_stackbox
unitree_z1_dual_arm_stackbox_v2
unitree_z1_dual_arm_cleanup_pencils
unitree_g1_pack_camera
```

## 🚀 Running Inference

Open:

```text
scripts/run_world_model_interaction.sh
```

Set the checkpoint path directly in this shell script. For example:

```bash
ckpt="/path/to/UnifoLM-WMA-0-Dual"
```

The script uses `configs/inference/world_model_interaction.yaml` as its inference configuration. No changes to this YAML file are required for the standard inference test.

Also configure the following variables in the same shell script:

```text
ckpt
res_dir
prompt_dir
datasets
```

The script uses GPU 0 by default:

```bash
CUDA_VISIBLE_DEVICES=0
```

For multi-GPU inference, change this value to the IDs of the GPUs you want to use, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1
```

Launch the inference script with:

```bash
bash scripts/run_world_model_interaction.sh
```

SwiftWMA integrates its inference optimizations into the original UnifoLM-WMA-0 interaction pipeline.

## 📝 Citation

If you find SwiftWMA useful in your research, please cite our paper:

```bibtex
@inproceedings{swiftwma,
  title     = {SwiftWMA: Cross-UNet Caching and Efficient Parallelism for World-Model-Action Inference},
  author    = {Haobo He and Qihang Chen and Zihan Tang and Zijian Li and Jiake Wang and Hao Hu and Haigang Zheng and Shiyi Li},
  booktitle = {IEEE International Conference on Computer Design (ICCD)},
  year      = {2026}
}
```
