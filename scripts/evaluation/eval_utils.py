import torch
import warnings
import torchvision
import sys
import pyarrow as pa
import logging
import concurrent.futures
import torch.multiprocessing as mp

from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Deque, Mapping, Union
from datasets.features.features import register_feature
from torch.utils.tensorboard.writer import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@dataclass
class VideoFrame:
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({
        "path": pa.string(),
        "timestamp": pa.float32()
    })
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    register_feature(VideoFrame, "VideoFrame")


def populate_queues(
        queues: Dict[str, Deque[Any]],
        batch: Mapping[str, Any]) -> Dict[str, Deque[Any]]:

    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues

def save_results(video: torch.Tensor, filename: str, fps: int = 8) -> None:
    """Save video tensor to file using torchvision.

    Args:
        video (Tensor): Tensor of shape (B, C, T, H, W).
        filename (str): Output file path.
        fps (int, optional): Frames per second. Defaults to 8.
    """
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename,
                               grid,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})

def save_results_async(video: torch.Tensor, filename: str, fps: int = 8, executor: concurrent.futures.Executor = None):
    """
    Args:
        video (Tensor): Tensor of shape (B, C, T, H, W).
        filename (str): Output file path.
        fps (int, optional): Frames per second. Defaults to 8.
        executor: 线程池执行器，可选。
    """
    def _save():
        v = video.detach().cpu().clone().share_memory_()
        v = torch.clamp(v.float(), -1., 1.)
        n = v.shape[0]
        v = v.permute(2, 0, 1, 3, 4)
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
            for framesheet in v
        ]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(filename,
                                grid,
                                fps=fps,
                                video_codec='h264',
                                options={'crf': '10'})
    executor.submit(_save)

def log_to_tensorboard(
        writer: SummaryWriter,
        data: Union[torch.Tensor, Any],
        tag: str,
        fps: int = 10) -> None:
    if isinstance(data, torch.Tensor) and data.dim() == 5:
        video = data
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = grid.unsqueeze(dim=0)
        writer.add_video(tag, grid, fps=fps)

def log_to_tensorboard_async(
        writer: SummaryWriter,
        data: Union[torch.Tensor, Any],
        tag: str,
        fps: int = 10,
        executor: concurrent.futures.Executor = None) -> None:
    def _log():
        if isinstance(data, torch.Tensor) and data.dim() == 5:
            video = data.detach().cpu().clone().share_memory_()
            n = video.shape[0]
            video = video.permute(2, 0, 1, 3, 4)
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]
            grid = torch.stack(frame_grids, dim=0)
            grid = (grid + 1.0) / 2.0
            grid = grid.unsqueeze(dim=0)
            writer.add_video(tag, grid, fps=fps)
    executor.submit(_log)
    
def save_and_log_sim_async(writer, video, tag, filename, fps=10, executor=None):
    """
    异步保存视频到磁盘并写入TensorBoard（普通模式）。
    """
    exec_obj = executor
    def _process():
        # Perform a single GPU->CPU transfer here and reuse it for logging, saving, and caller aggregation.
        v = video.detach().cpu()
        v_clamp = torch.clamp(v.float(), -1., 1.)
        n = v_clamp.shape[0]
        v_permute = v_clamp.permute(2, 0, 1, 3, 4)
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in v_permute]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        
        # 写入TensorBoard
        writer.add_video(tag, grid.unsqueeze(dim=0), fps=fps)
        
        # 保存到磁盘
        grid_uint8 = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(filename, grid_uint8, fps=fps, video_codec='h264', options={'crf': '20'})
        return v
    return exec_obj.submit(_process)

def save_and_log_dm_async(writer, model, latent, tag, filename, fps=10, executor=None):
    """
    异步decode后保存视频到磁盘并写入TensorBoard（DM模式）。
    """
    exec_obj = executor
    def _process():
        with torch.inference_mode():
            video = model.decode_first_stage(latent)
        v = video.detach().cpu()
        v_clamp = torch.clamp(v.float(), -1., 1.)
        n = v_clamp.shape[0]
        v_permute = v_clamp.permute(2, 0, 1, 3, 4)
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in v_permute]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        
        # 写入TensorBoard
        writer.add_video(tag, grid.unsqueeze(dim=0), fps=fps)
        
        # 保存到磁盘
        grid_uint8 = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(filename, grid_uint8, fps=fps, video_codec='h264', options={'crf': '20'})
    exec_obj.submit(_process)

class PostProcessExecutor:
    def __init__(self, num_workers=1):
        self.video_q = mp.Queue(maxsize=128)
        self.num_workers = num_workers
        self.pool = [mp.Process(target=self._worker, daemon=True) for _ in range(num_workers)]
        for p in self.pool:
            p.start()
        
    def _worker(self):
        tb_writer = None
        while True:
            item = self.video_q.get()
            if item is None:
                break
                
            if len(item) == 5:
                tag, v, filename, fps, tb_log_dir = item
            else:
                tag, v, filename, fps = item
                tb_log_dir = None
                
            if v.device.type != 'cpu':
                v = v.detach().cpu().clone().share_memory_()
                
            # Do the intensive processing
            v_clamp = torch.clamp(v.float(), -1., 1.)
            n = v_clamp.shape[0]
            v_permute = v_clamp.permute(2, 0, 1, 3, 4)
            frame_grids = [torchvision.utils.make_grid(fs, nrow=int(n), padding=0) for fs in v_permute]
            grid = torch.stack(frame_grids, dim=0)
            grid = (grid + 1.0) / 2.0
            
            # Save to disk
            if filename is not None:
                grid_uint8 = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
                torchvision.io.write_video(filename, grid_uint8, fps=fps, video_codec='h264', options={'crf': '20'})
                
            # Save to TensorBoard
            if tb_log_dir is not None:
                if tb_writer is None:
                    tb_writer = SummaryWriter(log_dir=tb_log_dir)
                tb_writer.add_video(tag, grid.unsqueeze(dim=0), fps=fps)

        if tb_writer is not None:
            tb_writer.close()
                
    def save_and_log(self, tag, v, filename, fps, tb_log_dir=None):
        self.video_q.put((tag, v, filename, fps, tb_log_dir))

    def close(self):
        for _ in range(self.num_workers):
            self.video_q.put(None)
        for p in self.pool:
            p.join()


def save_and_log_dm_async1(log_dir, model, latent, tag, filename, fps=10, executor=None, excutor1=None):
    mp_executor_global = executor
    def decode_and_save():
        with torch.inference_mode():
            video = model.decode_first_stage(latent)
        mp_executor_global.save_and_log(tag, video, filename, fps, tb_log_dir=log_dir)
    excutor1.submit(decode_and_save)

def save_and_log_sim_async1(log_dir, video, tag, filename, fps=10, executor=None):
    mp_executor_global = executor
    mp_executor_global.save_and_log(tag, video, filename, fps, tb_log_dir=log_dir)

