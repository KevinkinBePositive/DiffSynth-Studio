import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
input_image = Image.open("./zero_shot_test/Low-light_enhancement/low_light/780.png").resize((832, 480))

video = pipe(
    prompt="A static pixel-aligned visual transformation of the input low-light image into a well-lit enhanced image. The camera is completely fixed, producing a still scene with no motion. Every object, boundary, shape, texture, and layout element stays at the exact same image coordinates throughout the video. No object motion, no camera movement, no parallax, no zoom, no pan, no tilt, no rotation, no viewpoint change. Across time, only the visual appearance changes: the image gradually becomes brighter and clearer, restoring natural illumination, visibility, contrast, and color balance. Preserve the exact geometry, positions, proportions, object boundaries, scene layout, identity, and details of the input image. Do not add, remove, move, reshape, or redraw any object. The final frame shows the same scene as a clean low-light enhanced result: brighter exposure, improved shadow detail, reduced darkness, natural colors, and balanced contrast, while keeping the original composition and all objects perfectly aligned. Same scene, same coordinates, same geometry; only lighting and visibility improve. No flicker, no artifacts, no overexposure, no color distortion, no hallucinated details, no camera motion.",
    negative_prompt="motion, moving objects, walking, drifting, displacement, camera movement, camera pan, camera tilt, camera zoom, dolly, handheld camera, shake, parallax, viewpoint change, perspective change, object translation, object rotation, scene movement, background movement.",
    seed=27, tiled=True,
    height=480, width=832,
    num_frames=81,
    cfg_scale=4.0, 
    input_image=input_image,
    switch_DiT_boundary=0.9,
)
save_video(video, "./zero_shot_test/Low-light_enhancement/output/780_cfg4.mp4", fps=15, quality=9)
