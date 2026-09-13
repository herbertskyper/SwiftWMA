import argparse, os, glob
import time
import pandas as pd
import random
import torch
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
import imageio
import torch.distributed as dist
import torch.multiprocessing as mp

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from torch import nn
from eval_utils import *
from collections import deque
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch.profiler

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config

import os,concurrent.futures
import time

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
mp_executor = None

DEFAULT_FULL_MODEL_DIR = "/data/asc2026_data/T3/model"
DEFAULT_FULL_MODEL_FILE = os.path.join(DEFAULT_FULL_MODEL_DIR,
                                       "unifolm_wma_full_model.pt")


def resolve_full_model_path(path: str | None) -> str:
    if path is None:
        return DEFAULT_FULL_MODEL_FILE
    if path.endswith(".pt") or path.endswith(".pth"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "unifolm_wma_full_model.pt")


def load_full_model(path: str, map_location: str = "cpu") -> nn.Module:
    try:
        return torch.load(path,
                          map_location=map_location,
                          weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_infer_dtype(use_data: str) -> torch.dtype:
    use_data = use_data.lower()
    if use_data == "bf16":
        return torch.bfloat16
    if use_data == "fp16":
        return torch.float16
    return torch.float32


def cast_float_tensors(data: dict, dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return data
    casted = {}
    for key, value in data.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            casted[key] = value.to(dtype=dtype)
        else:
            casted[key] = value
    return casted


def init_distributed() -> tuple[int, int, int]:
    if dist.is_available() and not dist.is_initialized():
        env_world_size = os.environ.get("WORLD_SIZE")
        world_size = int(env_world_size)
        print(
            f'>>> Initializing distributed training ... world_size = {world_size}',
            flush=True,
        )
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _infer_world_size_from_visible_devices() -> int:
    if not torch.cuda.is_available():
        return 1
    return torch.cuda.device_count()


def _spawn_worker(local_rank: int, use_data: str, args: argparse.Namespace, world_size: int) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    seed_everything(seed)
    rank, gpu_num, local_rank = init_distributed()
    print(f'>>>>>>>>>>>>> Using data type: {use_data} ...')
    run_inference(args, gpu_num, local_rank, use_data)
    
    if mp_executor is not None:
        mp_executor.close()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): The model whose device is to be inferred.

    Returns:
        torch.device: The device of the model's parameters.
    """
    return next(iter(module.parameters())).device


def write_video(video_path: str, stacked_frames: list, fps: int) -> None:
    """Save a list of frames to a video file.

    Args:
        video_path (str): Output path for the video.
        stacked_frames (list): List of image frames.
        fps (int): Frames per second for the video.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "pkg_resources is deprecated as an API",
                                category=DeprecationWarning)
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    """Return sorted list of files in a directory matching specified postfixes.

    Args:
        data_dir (str): Directory path to search in.
        postfixes (list[str]): List of file extensions to match.

    Returns:
        list[str]: Sorted list of file paths.
    """
    patterns = [
        os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes
    ]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(model: nn.Module, ckpt, use_data: str = "fp32") -> nn.Module:
    """Load model weights from checkpoint file.

    Args:
        model (nn.Module): Model instance.
        ckpt (str or dict): Path to the checkpoint file OR loaded state_dict.

    Returns:
        nn.Module: Model with loaded weights.
    """
    model.to(f"cuda:{torch.cuda.current_device()}", non_blocking=True)
    if isinstance(ckpt, str):
        state_dict = torch.load(ckpt, map_location=f"cuda:{torch.cuda.current_device()}")
    else:
        state_dict = ckpt
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=False)
    else:
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def is_inferenced(save_dir: str, filename: str) -> bool:
    """Check if a given filename has already been processed and saved.

    Args:
        save_dir (str): Directory where results are saved.
        filename (str): Name of the file to check.

    Returns:
        bool: True if processed file exists, False otherwise.
    """
    video_file = os.path.join(save_dir, "samples_separate",
                              f"{filename[:-4]}_sample0.mp4")
    return os.path.exists(video_file)

def get_init_frame_path(data_dir: str, sample: dict) -> str:
    """Construct the init_frame path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing videos.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the video file.
    """
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    """Construct the full transition file path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing transition files.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the HDF5 transition file.
    """
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx: int,
                       init_frame_path: str,
                       transition_dict: dict[str, torch.Tensor],
                       frame_stride: int,
                       wma_data,
                       video_length: int = 16,
                       n_obs_steps: int = 2) -> dict[str, Tensor]:
    """
    Extracts a structured sample from a video sequence including frames, states, and actions,
    along with properly padded observations and pre-processed tensors for model input.

    Args:
        start_idx (int): Starting frame index for the current clip.
        video: decord video instance.
        transition_dict (Dict[str, Tensor]): Dictionary containing tensors for 'action', 
                                             'observation.state', 'action_type', 'state_type'.
        frame_stride (int): Temporal stride between sampled frames.
        wma_data: Object that holds configuration and utility functions like normalization, 
                transformation, and resolution info.
        video_length (int, optional): Number of frames to sample from the video. Default is 16.
        n_obs_steps (int, optional): Number of historical steps for observations. Default is 2.
    """

    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    """
    Extracts latent features from a video batch using the model's first-stage encoder.

    Args:
        model: the world model.
        videos (Tensor): Input videos of shape [B, C, T, H, W].

    Returns:
        Tensor: Latent video tensor of shape [B, C, T, H, W].
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def preprocess_observation(
        model, observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # Map to expected inputs for the policy
    return_observations = {}

    if isinstance(observations["pixels"], dict):
        imgs = {
            f"observation.images.{key}": img
            for key, img in observations["pixels"].items()
        }
    else:
        imgs = {"observation.images.top": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)

        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)

        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["agent_pos"]).float()
    return_observations['observation.state'] = model.normalize_inputs({
        'observation.state':
        return_observations['observation.state'].to(model.device)
    })['observation.state']

    return return_observations


def image_guided_synthesis_sim_mode(
        model: torch.nn.Module,
        prompts: list[str],
        observation: dict,
        noise_shape: tuple[int, int, int, int, int],
        action_cond_step: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int | None = None,
        text_input: bool = True,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
        sim_mode: bool = True,
    args: argparse.Namespace | None = None,
        **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs image-guided video generation in a simulation-style mode with optional multimodal guidance (image, state, action, text).

    Args:
        model (torch.nn.Module): The diffusion-based generative model with multimodal conditioning.
        prompts (list[str]): A list of textual prompts to guide the synthesis process.
        observation (dict): A dictionary containing observed inputs including:
            - 'observation.images.top': Tensor of shape [B, O, C, H, W] (top-down images)
            - 'observation.state': Tensor of shape [B, O, D] (state vector)
            - 'action': Tensor of shape [B, T, D] (action sequence)
        noise_shape (tuple[int, int, int, int, int]): Shape of the latent variable to generate, 
            typically (B, C, T, H, W).
        action_cond_step (int): Number of time steps where action conditioning is applied. Default is 16.
        n_samples (int): Number of samples to generate (unused here, always generates 1). Default is 1.
        ddim_steps (int): Number of DDIM sampling steps. Default is 50.
        ddim_eta (float): DDIM eta parameter controlling the stochasticity. Default is 1.0.
        unconditional_guidance_scale (float): Scale for classifier-free guidance. If 1.0, guidance is off.
        fs (int | None): Frame index to condition on, broadcasted across the batch if specified. Default is None.
        text_input (bool): Whether to use text prompt as conditioning. If False, uses empty strings. Default is True.
        timestep_spacing (str): Timestep sampling method in DDIM sampler. Typically "uniform" or "linspace".
        guidance_rescale (float): Guidance rescaling factor to mitigate overexposure from classifier-free guidance.
        sim_mode (bool): Whether to perform world-model interaction or decision-making using the world-model.
        **kwargs: Additional arguments passed to the DDIM sampler.

    Returns:
        batch_variants (torch.Tensor): Predicted pixel-space video frames [B, C, T, H, W].
        actions (torch.Tensor): Predicted action sequences [B, T, D] from diffusion decoding.
        states (torch.Tensor): Predicted state sequences [B, T, D] from diffusion decoding.
    """
    b, _, t, _, _ = noise_shape
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]

    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
    cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
    cond_img_emb = model.embedder(cond_img)
    cond_img_emb = model.image_proj_model(cond_img_emb)

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
        img_cat_cond = z[:, :, -1:, :, :]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=noise_shape[2])
        cond = {"c_concat": [img_cat_cond]}

    if not text_input:
        prompts = [""] * batch_size
    cond_ins_emb = model.get_learned_conditioning(prompts)

    cond_state_emb = model.state_projector(observation['observation.state'])
    cond_state_emb = cond_state_emb + model.agent_state_pos_emb

    cond_action_emb = model.action_projector(observation['action'])
    cond_action_emb = cond_action_emb + model.agent_action_pos_emb

    if not sim_mode:
        cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat(
            [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
            dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'][:, :,
                                              -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:],
        # sim_mode,
        # False,
    ]

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    if sim_mode == False:
        kwargs["sim_mode"] = False
    else:
        kwargs["sim_mode"] = True
    if args is not None and args.use_deepcache:
        kwargs["use_deepcache"] = True
        kwargs["deepcache_interval"] = 9
        kwargs["deepcache_refresh_step"] = 0
        kwargs["deepcache_warmup_steps"] = 0
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        samples, actions, states, intermedia = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs)

        if kwargs["sim_mode"]:
            # Reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants = batch_images
        else:
            batch_variants = samples

    return batch_variants, actions, states


def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int, use_data: str) -> None:
    """
    Run inference pipeline on prompts and image inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        gpu_num (int): Number of GPUs.
        gpu_no (int): Index of the current GPU.

    Returns:
        None
    """
    import time
    time1 = time.time()
    global mp_executor
    # Create inference and tensorboard dirs
    log_dir = None
    if is_main_process():
        os.makedirs(args.savedir + '/inference', exist_ok=True)
        log_dir = args.savedir + f"/tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        # writer = SummaryWriter(log_dir=log_dir) if not args.no_log else None

    if not args.no_log and is_main_process():
        mp_executor = PostProcessExecutor()
    else:
        mp_executor = None

    # Load prompt
    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    # Load config
    config = OmegaConf.load(args.config)
    config['model']['params']['wma_config']['params'][
        'use_checkpoint'] = False
    # model = instantiate_from_config(config.model)
    # model.perframe_ae = args.perframe_ae
    # assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    # model = load_model_checkpoint(model, args.ckpt_path)

    # model.eval()
    # print(f'>>> Load pre-trained model ...')
    full_model_path = resolve_full_model_path(None)
    model = None

    if args.load_full_model:
        assert os.path.exists(full_model_path), f"Error: full model file not found at {full_model_path}!"
        print(f'>>> Loading full model object from: {full_model_path}', flush=True)
        model = load_full_model(full_model_path, map_location='cpu')
        model.perframe_ae = args.perframe_ae
        print(f'>>> Full model loading time: {time.time() - time1:.2f} seconds', flush=True)
    else:
        instantiate_future = executor.submit(instantiate_from_config, config.model)
        def _load_ckpt_to_cpu(path):
            assert os.path.exists(path), "Error: checkpoint Not Found!"
            sd = torch.load(path, map_location='cpu')
            return sd
        ckpt_future = executor.submit(_load_ckpt_to_cpu, args.ckpt_path)
    # # 检查模型参数
    # for name, param in model.named_parameters():
    #     print(f'Parameter name: {name}, shape: {param.shape}')
    # # 检查模型使用的数据类型
    # for name, buffer in model.named_buffers():
    #     print(f'Buffer name: {name}, shape: {buffer.shape}, dtype: {buffer.dtype}')

    # Build unnomalizer
    logging.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    print(">>> Dataset is successfully loaded ...")

    if model is None:
        print(">>> Main: waiting for model instantiation / ckpt load to complete ...")
        model = instantiate_future.result()
        model.perframe_ae = args.perframe_ae
        sd = ckpt_future.result()

        # normalize state_dict shape as original load_model_checkpoint did
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        else:
            if "module" in sd:
                new_pl_sd = OrderedDict()
                for key in sd['module'].keys():
                    new_pl_sd[key[16:]] = sd['module'][key]
                sd = new_pl_sd

        # try loading, with fallback for key rename used in older checkpoints
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            new_pl_sd = OrderedDict()
            for k, v in sd.items():
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = v
                else:
                    new_pl_sd[k] = v
            model.load_state_dict(new_pl_sd, strict=False)
        # print('>>> model checkpoint loaded.')
        print('>>> Main prework done.')
        print(f'>>> Model instantiation and checkpoint loading time: {time.time() - time1:.2f} seconds', flush=True)

        if is_main_process():
            torch.save(model, full_model_path)
            print(f'>>> Full model object exported to: {full_model_path}', flush=True)
            # assert False, ">>> Full model saved, exiting to free up GPU memory. Please run the script again to perform inference with the full model loaded from disk."

    # Move to GPU (this triggers CUDA init / allocation)
    model.to(f"cuda:{gpu_no}", non_blocking=True)
    infer_dtype = get_infer_dtype(use_data)
    if infer_dtype != torch.float32:
        model = model.to(dtype=infer_dtype)
        print(f'>>> Convert model to dtype: {infer_dtype}', flush=True)
    model.eval()

    # Configure MACs measurement on current rank.
    macs_on_this_rank = False
    if args.measure_macs:
        macs_on_this_rank = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        # In strict MACs mode, all ranks must follow the same profiled path to avoid DDP collective mismatches.
        model.model.macs_count_enabled = True
        if macs_on_this_rank:
            print(
                f">>> MACs measurement enabled on rank {dist.get_rank() if dist.is_available() and dist.is_initialized() else 0} (counting + logging).",
                flush=True,
            )
        elif dist.is_available() and dist.is_initialized():
            print(f">>> MACs measurement enabled on rank {dist.get_rank()} (execution only; rank 0 counts/logs).", flush=True)
    else:
        model.model.macs_count_enabled = False
    
    # Handle TRT Export / Load
    if hasattr(args, "export_onnx") and args.export_onnx:
        model.model.export_onnx_path = args.export_onnx
        # model.eval()
        # # Call export immediately using dummy inputs generated inside the method
        # model.model.export_onnx(args.export_onnx)
        # # export_onnx will exit, but just in case:
        # import sys; sys.exit(0)
         
    elif hasattr(args, "use_trt") and args.use_trt:
        engine_file = "/data/asc2026_data/T3/trt/unifolm_wma.engine"
        # Try same dir fallback
        if not os.path.exists(engine_file):
            base_dir = os.path.dirname(args.ckpt_path)
            candidates = [f for f in os.listdir(base_dir) if f.endswith(".engine")]
            if candidates:
                engine_file = os.path.join(base_dir, candidates[0])
        model.model.enable_trt(engine_file)

    # model = model.cuda(gpu_no)

    if use_data in ("tf32"):
        if torch.__version__ >= "2" and torch.cuda.is_available():
            # print(">>> Use torch compile to accelerate the model ...")
            torch.set_float32_matmul_precision('high')
            # if use_data == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # model = torch.compile(model, mode="reduce-overhead")
            # arch_file = os.path.join(args.savedir, 'model_architecture.txt')
            # print_model_architecture(model, arch_file)
            # exit(0)
            # compile_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # compile_future = compile_executor.submit(torch.compile, model, mode="reduce-overhead")
            # compile_future = executor.submit(torch.compile, model, mode="reduce-overhead")

    device = get_device_from_parameters(model)

    # Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # Start inference
    for idx in range(0, len(df)):
        sample = df.iloc[idx]

        # Got initial frame path
        init_frame_path = get_init_frame_path(args.prompt_dir, sample)
        ori_fps = float(sample['fps'])

        video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(video_save_dir + '/dm', exist_ok=True)
        os.makedirs(video_save_dir + '/wm', exist_ok=True)

        # Load transitions to get the initial state later
        transition_path = get_transition_path(args.prompt_dir, sample)
        with h5py.File(transition_path, 'r') as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]

        # If many, test various frequence control and world-model generation
        for fs in args.frame_stride:

            # For saving imagens in policy
            sample_save_dir = f'{video_save_dir}/dm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For saving environmental changes in world-model
            sample_save_dir = f'{video_save_dir}/wm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For collecting interaction videos
            wm_video = []
            # Initialize observation queues
            cond_obs_queues = {
                "observation.images.top":
                deque(maxlen=model.n_obs_steps_imagen),
                "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                "action": deque(maxlen=args.video_length),
            }
            # Obtain initial frame and state
            start_idx = 0
            model_input_fs = ori_fps // fs
            batch, ori_state_dim, ori_action_dim = prepare_init_input(
                start_idx,
                init_frame_path,
                transition_dict,
                fs,
                data.test_datasets[args.dataset],
                n_obs_steps=model.n_obs_steps_imagen)
            observation = {
                'observation.images.top':
                batch['observation.image'].permute(1, 0, 2,
                                                   3)[-1].unsqueeze(0),
                'observation.state':
                batch['observation.state'][-1].unsqueeze(0),
                'action':
                torch.zeros_like(batch['action'][-1]).unsqueeze(0)
            }
            observation = {
                key: observation[key].to(device, non_blocking=True)
                for key in observation
            }
            observation = cast_float_tensors(observation, infer_dtype)
            # Update observation queues
            cond_obs_queues = populate_queues(cond_obs_queues, observation)
            time1 = time.time()
            # Multi-round interaction with the world-model
            for itr in tqdm(range(args.n_iter)):

                # Get observation
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }
                observation = cast_float_tensors(observation, infer_dtype)

                # Use world-model in policy to generate action
                print(f'>>> Step {itr}: generating actions ...', flush=True)
                # if itr == 0 and use_data in ["bf16", "tf32"]:
                #     ii = 0
                #     while not compile_future.done():
                #         time.sleep(0.1)
                #         ii = ii + 1
                #     model = compile_future.result()
                #     print(f'>>> Model compile waiting time: {ii * 0.1:.2f} seconds')
                # do_profile = (itr == 1)
                # prof_dm = None
                # if do_profile:
                #     prof_dm = torch.profiler.profile(
                #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                #         record_shapes=True,
                #         with_stack=True,
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.savedir, 'profiler_logs', 'dm'))
                #     )
                #     prof_dm.start()
                with torch.inference_mode():
                    pred_videos_0, pred_actions, _ = image_guided_synthesis_sim_mode(
                        model,
                        sample['instruction'],
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=args.ddim_steps,
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale,
                        sim_mode=False,
                        args=args)

                # if prof_dm is not None:
                #     prof_dm.stop()

                # Update future actions in the observation queues
                for idx in range(len(pred_actions[0])):
                    observation = {'action': pred_actions[0][idx:idx + 1].clone()}
                    observation['action'][:, ori_action_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                    observation)

                # Collect data for interacting the world-model using the predicted actions
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }
                observation = cast_float_tensors(observation, infer_dtype)
                if not args.no_log and is_main_process():
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-dm-fs-{fs}/itr-{itr}"
                    sample_video_file = f'{video_save_dir}/dm/{fs}/itr-{itr}.mp4'
                    save_and_log_dm_async1(
                        log_dir,
                        model,
                        pred_videos_0,      # latent
                        sample_tag,
                        sample_video_file,
                        fps=args.save_fps,
                        executor=mp_executor,
                        excutor1=executor
                    )

                # Interaction with the world-model
                print(f'>>> Step {itr}: interacting with world model ...')
                if args.first_itr_fp32 and itr == 0:
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        with torch.inference_mode():
                            pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                                model,
                                "",
                                observation,
                                noise_shape,
                                action_cond_step=args.exe_steps,
                                ddim_steps=args.ddim_steps,
                                ddim_eta=args.ddim_eta,
                                unconditional_guidance_scale=args.
                                unconditional_guidance_scale,
                                fs=model_input_fs,
                                text_input=False,
                                timestep_spacing=args.timestep_spacing,
                                guidance_rescale=args.guidance_rescale,
                                args=args)
                else:
                    with torch.inference_mode():
                        pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                            model,
                            "",
                            observation,
                            noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.
                            unconditional_guidance_scale,
                            fs=model_input_fs,
                            text_input=False,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            args=args)

                for idx in range(args.exe_steps):
                    observation = {
                        'observation.images.top':
                        pred_videos_1[0][:, idx:idx + 1].permute(1, 0, 2, 3),
                        'observation.state':
                        torch.zeros_like(pred_states[0][idx:idx + 1]) if
                        args.zero_pred_state else pred_states[0][idx:idx + 1].clone(),
                        'action':
                        torch.zeros_like(pred_actions[0][-1:])
                    }
                    observation['observation.state'][:, ori_state_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                    observation)

                print('>' * 24)
                if not args.no_log and is_main_process():
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/itr-{itr}"
                    sample_video_file = f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4'
                    save_and_log_sim_async1(
                        log_dir,
                        pred_videos_1[:, :, :args.exe_steps],
                        sample_tag,
                        sample_video_file,
                        fps=args.save_fps,
                        executor=mp_executor
                    )
                    wm_video.append(executor.submit(lambda t: t.cpu(), pred_videos_1[:, :, :args.exe_steps]))
                elif is_main_process():
                    wm_video.append(executor.submit(lambda t: t.cpu(), pred_videos_1[:, :, :args.exe_steps]))       
            
            if is_main_process():
                print(f"Inference for sample {sample['videoid']} with frame stride {fs} completed in {time.time() - time1:.2f} seconds.")
                if args.measure_macs and macs_on_this_rank:
                    measured_macs = model.model.get_total_macs()
                    total_macs = measured_macs * gpu_num
                    print(
                        f">>> MACs for sample {sample['videoid']} fs {fs}: {total_macs / 1e12:.6f} T (single-rank measured {measured_macs / 1e12:.6f} T x {gpu_num} GPUs)",
                        flush=True,
                    )
                resolved_video = [v.result() if hasattr(v, 'result') else v for v in wm_video]
                full_video = torch.cat(resolved_video, dim=2)
                sample_full_video_file = f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4"
                if not args.no_log:
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/full"
                    mp_executor.save_and_log(sample_tag, full_video, sample_full_video_file, args.save_fps, tb_log_dir=log_dir)
                else:
                    save_results(full_video, sample_full_video_file, fps=args.save_fps)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="the name of dataset to test")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs='+',
        required=True,
        help=
        "frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)"
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument("--num_generation",
                        type=int,
                        default=1,
                        help="seed for seed_everything")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help=
        "Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help=
        "Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help=
        "Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024."
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model",
    )
    parser.add_argument("--zero_pred_state",
                        action='store_true',
                        default=False,
                        help="not using the predicted states as comparison")
    parser.add_argument("--save_fps",
                        type=int,
                        default=8,
                        help="fps for the saving video")
    parser.add_argument("--export_onnx",
                        type=str,
                        default=None,
                        help="Path to export ONNX model (e.g. model.onnx)")
    parser.add_argument("--use_trt",
                        action='store_true',
                        default=False,
                        help="Use TensorRT engine (looks for .engine file)")
    parser.add_argument("--use_deepcache",
                        action='store_true',
                        default=True,
                        help="Enable DeepCache acceleration.")
    parser.add_argument("--use_data",
                        type=str,
                        default="tf32",
                        choices=["fp32", "fp16", "tf32"],
                        help="Inference precision mode.")
    parser.add_argument(
        "--load_full_model",
        action='store_true',
        default=True,
        help=
        "Load full model object directly via torch.load(path) and skip instantiate_from_config + checkpoint parsing."
    )
    parser.add_argument("--no_log",
                    default=False,
                    help="Disable logging output")
    parser.add_argument("--first_itr_fp32",
                    action='store_true',
                    default=False,
                    help="Use FP32 only for the first iteration (overrides use_data for first itr).")
    parser.add_argument("--multi_gpu",
                        action='store_true',
                        default=True,
                        help="Enable multi-GPU distributed spawn (overrides auto-detection).")
    parser.add_argument("--measure_macs",
                        action='store_true',
                        default=True,
                        help="Measure full inference MACs using thop and print the result in T.")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    use_data = args.use_data
    world_size = _infer_world_size_from_visible_devices()
    if world_size > 1 and not args.use_trt and not args.export_onnx and args.multi_gpu:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        mp.spawn(_spawn_worker, args=(use_data, args, world_size), nprocs=world_size, join=True)
    else:
        seed = args.seed
        if seed < 0:
            seed = random.randint(0, 2**31)
        seed_everything(seed)
        rank, gpu_num = 0, 1
        print(f'>>>>>>>>>>>>> Using data type: {use_data} ...')
        run_inference(args, gpu_num, rank, use_data)
        
        if mp_executor is not None:
            mp_executor.close()