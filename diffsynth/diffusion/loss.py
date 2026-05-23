from .base_pipeline import BasePipeline
import torch
import torch.nn.functional as F


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


def _flow_match_sigma(pipe: BasePipeline, timestep: torch.Tensor, sample: torch.Tensor):
    timestep_id = torch.argmin((pipe.scheduler.timesteps - timestep.to(pipe.scheduler.timesteps.device)).abs())
    sigma = pipe.scheduler.sigmas[timestep_id]
    return sigma.to(device=sample.device, dtype=sample.dtype)


def _flow_match_pred_x0(pipe: BasePipeline, noisy_latents: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor):
    sigma = _flow_match_sigma(pipe, timestep, noisy_latents)
    while sigma.ndim < noisy_latents.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return noisy_latents - sigma * model_output


def _wan_iteration_models(pipe: BasePipeline, source_pipe: BasePipeline, timestep: torch.Tensor, switch_DiT_boundary: float):
    timestep_value = float(timestep.detach().float().mean().item())
    use_second_dit = timestep_value < switch_DiT_boundary * 1000 and getattr(source_pipe, "dit2", None) is not None
    model_names = source_pipe.in_iteration_models_2 if use_second_dit else source_pipe.in_iteration_models
    models = {name: getattr(source_pipe, name) for name in model_names}
    if use_second_dit:
        models["dit"] = models.pop("dit2")
        if "vace2" in models:
            models["vace"] = models.pop("vace2")
    return models


def _wan_predict(
    pipe: BasePipeline,
    source_pipe: BasePipeline,
    inputs_shared,
    inputs_posi,
    inputs_nega,
    timestep,
    cfg_scale=1.0,
    switch_DiT_boundary=0.9,
    progress_id=None,
):
    models = _wan_iteration_models(pipe, source_pipe, timestep, switch_DiT_boundary)
    if cfg_scale != 1.0:
        return pipe.cfg_guided_model_fn(
            pipe.model_fn, cfg_scale,
            inputs_shared, inputs_posi, inputs_nega,
            **models, timestep=timestep, progress_id=progress_id
        )
    return pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)


def DMD2FlowMatchLoss(
    pipe: BasePipeline,
    teacher_pipe: BasePipeline,
    guidance_pipe: BasePipeline,
    inputs_shared,
    inputs_posi,
    inputs_nega,
    num_inference_steps=8,
    student_cfg_scale=1.0,
    real_guidance_scale=4.0,
    fake_guidance_scale=1.0,
    dm_loss_weight=1.0,
    fake_loss_weight=1.0,
    min_step_percent=0.02,
    max_step_percent=0.98,
    switch_DiT_boundary=0.9,
    sigma_shift=5.0,
):
    """DMD2-style distribution matching for Flow Matching video pipelines.

    This follows DMD2's two-score idea: a frozen teacher estimates the real
    score, a trainable fake-guidance model estimates the generator score, and
    the student is updated with the resulting distribution-matching gradient.
    """
    sample_inputs = dict(inputs_shared)
    sample_inputs["latents"] = sample_inputs["latents"].clone()

    pipe.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = _wan_predict(
            pipe, pipe, sample_inputs, inputs_posi, inputs_nega,
            timestep=timestep, cfg_scale=student_cfg_scale,
            switch_DiT_boundary=switch_DiT_boundary,
            progress_id=progress_id,
        )
        sample_inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **sample_inputs)

    generated_latents = sample_inputs["latents"]

    pipe.scheduler.set_timesteps(1000, training=True, shift=sigma_shift)
    min_step = int(min_step_percent * len(pipe.scheduler.timesteps))
    max_step = int(max_step_percent * len(pipe.scheduler.timesteps))
    max_step = max(min(max_step, len(pipe.scheduler.timesteps)), min_step + 1)
    timestep_id = torch.randint(min_step, max_step, (1,), device=generated_latents.device)
    timestep = pipe.scheduler.timesteps[timestep_id.cpu()].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(generated_latents)
    noisy_latents = pipe.scheduler.add_noise(generated_latents, noise, timestep)

    score_inputs = dict(sample_inputs)
    score_inputs["latents"] = noisy_latents

    with torch.no_grad():
        pred_fake = _wan_predict(
            pipe, guidance_pipe, score_inputs, inputs_posi, inputs_nega,
            timestep=timestep, cfg_scale=fake_guidance_scale,
            switch_DiT_boundary=switch_DiT_boundary,
        )
        pred_fake_x0 = _flow_match_pred_x0(pipe, noisy_latents, pred_fake, timestep)

        pred_real = _wan_predict(
            pipe, teacher_pipe, score_inputs, inputs_posi, inputs_nega,
            timestep=timestep, cfg_scale=real_guidance_scale,
            switch_DiT_boundary=switch_DiT_boundary,
        )
        pred_real_x0 = _flow_match_pred_x0(pipe, noisy_latents, pred_real, timestep)

        p_real = generated_latents - pred_real_x0
        p_fake = generated_latents - pred_fake_x0
        reduce_dims = tuple(range(1, generated_latents.ndim))
        grad = (p_real - p_fake) / torch.clamp(p_real.abs().mean(dim=reduce_dims, keepdim=True), min=1e-6)
        grad = torch.nan_to_num(grad)

    loss_dm = 0.5 * F.mse_loss(generated_latents.float(), (generated_latents - grad).detach().float())

    fake_latents = generated_latents.detach()
    fake_noise = torch.randn_like(fake_latents)
    fake_noisy_latents = pipe.scheduler.add_noise(fake_latents, fake_noise, timestep)
    fake_inputs = dict(sample_inputs)
    fake_inputs["latents"] = fake_noisy_latents
    pred_fake_train = _wan_predict(
        pipe, guidance_pipe, fake_inputs, inputs_posi, inputs_nega,
        timestep=timestep, cfg_scale=1.0,
        switch_DiT_boundary=switch_DiT_boundary,
    )
    fake_target = pipe.scheduler.training_target(fake_latents, fake_noise, timestep)
    loss_fake = F.mse_loss(pred_fake_train.float(), fake_target.float())

    return dm_loss_weight * loss_dm + fake_loss_weight * loss_fake


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
