import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
input_image = Image.open("./zero_shot_test/Segmentation/img/000000000063.jpg").resize((832, 480))

video = pipe(
    prompt="Create an animation of instance segmentation being performed on this photograph: each distinct entity is overlaid in a different flat color. Scene: The animation starts from the provided, unaltered photograph. The scene in the photograph is static and doesn’t move. First, the background fades to white. Then, the first entity is covered by a flat color, perfectly preserving its silhouette. Then the second entity, too, is covered by a different flat color, perfectly preserving its silhouette. One by one, each entity is covered by a different flat color. Finally, all entities are covered with different colors. Camera: Static shot without camera movement. No pan. No rotation. No zoom. No glitches or artifacts.",
    negative_prompt="",
    seed=27, tiled=True,
    height=480, width=832,
    num_frames=81,
    cfg_scale=5.0, 
    input_image=input_image,
    switch_DiT_boundary=0.9,
)
save_video(video, "./zero_shot_test/Segmentation/output/63.mp4", fps=15, quality=9)
