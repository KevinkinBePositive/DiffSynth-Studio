import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dmd2_teacher_model_id_with_origin_paths=None,
        dmd2_guidance_model_id_with_origin_paths=None,
        dmd2_guidance_lora_rank=32,
        dmd2_guidance_lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        dmd2_guidance_lora_checkpoint=None,
        dmd2_num_inference_steps=8,
        dmd2_student_cfg_scale=1.0,
        dmd2_real_guidance_scale=4.0,
        dmd2_fake_guidance_scale=1.0,
        dmd2_dm_loss_weight=1.0,
        dmd2_fake_loss_weight=1.0,
        dmd2_min_step_percent=0.02,
        dmd2_max_step_percent=0.98,
        dmd2_switch_DiT_boundary=0.9,
        dmd2_sigma_shift=5.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        if task.startswith("dmd2_distill"):
            self.setup_dmd2_models(
                model_id_with_origin_paths=model_id_with_origin_paths,
                teacher_model_id_with_origin_paths=dmd2_teacher_model_id_with_origin_paths,
                guidance_model_id_with_origin_paths=dmd2_guidance_model_id_with_origin_paths,
                guidance_lora_base_model=lora_base_model or (trainable_models.split(",")[0] if trainable_models is not None else "dit"),
                guidance_lora_rank=dmd2_guidance_lora_rank,
                guidance_lora_target_modules=dmd2_guidance_lora_target_modules,
                guidance_lora_checkpoint=dmd2_guidance_lora_checkpoint,
                device=device,
            )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.dmd2_num_inference_steps = dmd2_num_inference_steps
        self.dmd2_student_cfg_scale = dmd2_student_cfg_scale
        self.dmd2_real_guidance_scale = dmd2_real_guidance_scale
        self.dmd2_fake_guidance_scale = dmd2_fake_guidance_scale
        self.dmd2_dm_loss_weight = dmd2_dm_loss_weight
        self.dmd2_fake_loss_weight = dmd2_fake_loss_weight
        self.dmd2_min_step_percent = dmd2_min_step_percent
        self.dmd2_max_step_percent = dmd2_max_step_percent
        self.dmd2_switch_DiT_boundary = dmd2_switch_DiT_boundary
        self.dmd2_sigma_shift = dmd2_sigma_shift
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "dmd2_distill": self.dmd2_loss,
            "dmd2_distill:train": self.dmd2_loss,
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def setup_dmd2_models(
        self,
        model_id_with_origin_paths,
        teacher_model_id_with_origin_paths,
        guidance_model_id_with_origin_paths,
        guidance_lora_base_model,
        guidance_lora_rank,
        guidance_lora_target_modules,
        guidance_lora_checkpoint,
        device,
    ):
        teacher_paths = teacher_model_id_with_origin_paths or model_id_with_origin_paths
        guidance_paths = guidance_model_id_with_origin_paths or teacher_paths
        teacher_configs = self.parse_model_configs(None, teacher_paths, device=device)
        guidance_configs = self.parse_model_configs(None, guidance_paths, device=device)
        self.dmd2_teacher_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=teacher_configs, tokenizer_config=None, audio_processor_config=None,
        )
        self.dmd2_guidance_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=guidance_configs, tokenizer_config=None, audio_processor_config=None,
        )
        self.dmd2_teacher_pipe.freeze_except([])
        self.dmd2_guidance_pipe.freeze_except([])

        if guidance_lora_rank > 0:
            if not hasattr(self.dmd2_guidance_pipe, guidance_lora_base_model) or getattr(self.dmd2_guidance_pipe, guidance_lora_base_model) is None:
                raise ValueError(f"DMD2 guidance model `{guidance_lora_base_model}` is not available.")
            guidance_model = getattr(self.dmd2_guidance_pipe, guidance_lora_base_model)
            guidance_model = self.add_lora_to_model(
                guidance_model,
                target_modules=self.parse_lora_target_modules(guidance_model, guidance_lora_target_modules),
                lora_rank=guidance_lora_rank,
                upcast_dtype=self.dmd2_guidance_pipe.torch_dtype,
            )
            if guidance_lora_checkpoint is not None:
                lora = load_state_dict(guidance_lora_checkpoint)
                lora_loader = self.dmd2_guidance_pipe.lora_loader(torch_dtype=self.dmd2_guidance_pipe.torch_dtype, device=self.dmd2_guidance_pipe.device)
                lora = lora_loader.convert_state_dict(lora)
                lora = self.mapping_lora_state_dict(lora)
                guidance_model.load_state_dict(lora, strict=False)
            setattr(self.dmd2_guidance_pipe, guidance_lora_base_model, guidance_model)
        else:
            self.dmd2_guidance_pipe.freeze_except([guidance_lora_base_model])

    def dmd2_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        return DMD2FlowMatchLoss(
            pipe, self.dmd2_teacher_pipe, self.dmd2_guidance_pipe,
            inputs_shared, inputs_posi, inputs_nega,
            num_inference_steps=self.dmd2_num_inference_steps,
            student_cfg_scale=self.dmd2_student_cfg_scale,
            real_guidance_scale=self.dmd2_real_guidance_scale,
            fake_guidance_scale=self.dmd2_fake_guidance_scale,
            dm_loss_weight=self.dmd2_dm_loss_weight,
            fake_loss_weight=self.dmd2_fake_loss_weight,
            min_step_percent=self.dmd2_min_step_percent,
            max_step_percent=self.dmd2_max_step_percent,
            switch_DiT_boundary=self.dmd2_switch_DiT_boundary,
            sigma_shift=self.dmd2_sigma_shift,
        )

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        if self.task.startswith("dmd2_distill"):
            state_dict = {name: param for name, param in state_dict.items() if name.startswith("pipe.")}
        return super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--dmd2_teacher_model_id_with_origin_paths", type=str, default=None, help="Teacher model configs for DMD2. Defaults to --model_id_with_origin_paths.")
    parser.add_argument("--dmd2_guidance_model_id_with_origin_paths", type=str, default=None, help="Fake-score guidance model configs for DMD2. Defaults to teacher configs.")
    parser.add_argument("--dmd2_guidance_lora_rank", type=int, default=32, help="LoRA rank for the DMD2 fake-score guidance model. Set 0 for full guidance training.")
    parser.add_argument("--dmd2_guidance_lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="LoRA target modules for the DMD2 fake-score guidance model.")
    parser.add_argument("--dmd2_guidance_lora_checkpoint", type=str, default=None, help="Optional LoRA checkpoint for the DMD2 fake-score guidance model.")
    parser.add_argument("--dmd2_num_inference_steps", type=int, default=8, help="Number of student denoising steps used during DMD2 training.")
    parser.add_argument("--dmd2_student_cfg_scale", type=float, default=1.0, help="CFG scale used by the student generator during DMD2 sampling.")
    parser.add_argument("--dmd2_real_guidance_scale", type=float, default=4.0, help="CFG scale used by the frozen teacher score in DMD2.")
    parser.add_argument("--dmd2_fake_guidance_scale", type=float, default=1.0, help="CFG scale used by the fake-score model in DMD2.")
    parser.add_argument("--dmd2_dm_loss_weight", type=float, default=1.0, help="Weight for DMD2 distribution matching loss.")
    parser.add_argument("--dmd2_fake_loss_weight", type=float, default=1.0, help="Weight for DMD2 fake-score training loss.")
    parser.add_argument("--dmd2_min_step_percent", type=float, default=0.02, help="Minimum DMD2 score timestep as a scheduler-index fraction.")
    parser.add_argument("--dmd2_max_step_percent", type=float, default=0.98, help="Maximum DMD2 score timestep as a scheduler-index fraction.")
    parser.add_argument("--dmd2_switch_DiT_boundary", type=float, default=0.9, help="Wan2.2 high/low DiT switch boundary used while sampling the 8-step student.")
    parser.add_argument("--dmd2_sigma_shift", type=float, default=5.0, help="Wan FlowMatch sigma shift for DMD2 sampling and score timesteps.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    extra_inputs = set(args.extra_inputs.split(",")) if args.extra_inputs is not None else set()
    special_operator_map = {
        "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
        "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
    }
    if "input_audio" in extra_inputs:
        special_operator_map["input_audio"] = ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000)

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map=special_operator_map,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dmd2_teacher_model_id_with_origin_paths=args.dmd2_teacher_model_id_with_origin_paths,
        dmd2_guidance_model_id_with_origin_paths=args.dmd2_guidance_model_id_with_origin_paths,
        dmd2_guidance_lora_rank=args.dmd2_guidance_lora_rank,
        dmd2_guidance_lora_target_modules=args.dmd2_guidance_lora_target_modules,
        dmd2_guidance_lora_checkpoint=args.dmd2_guidance_lora_checkpoint,
        dmd2_num_inference_steps=args.dmd2_num_inference_steps,
        dmd2_student_cfg_scale=args.dmd2_student_cfg_scale,
        dmd2_real_guidance_scale=args.dmd2_real_guidance_scale,
        dmd2_fake_guidance_scale=args.dmd2_fake_guidance_scale,
        dmd2_dm_loss_weight=args.dmd2_dm_loss_weight,
        dmd2_fake_loss_weight=args.dmd2_fake_loss_weight,
        dmd2_min_step_percent=args.dmd2_min_step_percent,
        dmd2_max_step_percent=args.dmd2_max_step_percent,
        dmd2_switch_DiT_boundary=args.dmd2_switch_DiT_boundary,
        dmd2_sigma_shift=args.dmd2_sigma_shift,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "dmd2_distill": launch_training_task,
        "dmd2_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
