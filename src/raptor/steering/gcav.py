"""Approved GCAV kernels; function bodies retained from producer e3b924e.

Optional raw probes use their own W/B and W dot v calibration. Without raw
weights the original unit-direction probability convention is retained.
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch

def get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
        return model.model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unable to locate transformer blocks.")

def select_active_layers(
    vectors: Sequence[Optional[torch.Tensor]],
    validation_accuracies: Sequence[Optional[float]],
    *,
    layers: Optional[Sequence[int]] = None,
    accuracy_threshold: float = 0.9,
    reliability_enabled: Optional[bool] = None,
    reliability_comparison: Optional[str] = ">",
) -> list[int]:
    """Keep every reliable candidate layer, preserving the original range.

    This is threshold filtering for steering, separate from classification's
    single best validation layer. Explicit layer lists still require reliability.
    ``None`` preserves the legacy threshold>0 enabled convention. An explicit
    task policy can use inclusive comparison or disable filtering entirely.
    """
    if reliability_enabled is not None and not isinstance(reliability_enabled, bool):
        raise ValueError("reliability_enabled must be an explicit bool or legacy None")
    enabled = accuracy_threshold > 0 if reliability_enabled is None else reliability_enabled
    if reliability_comparison not in {">", ">=", None} or (enabled and reliability_comparison is None):
        raise ValueError("enabled reliability comparison must be > or >=")
    if enabled and (isinstance(accuracy_threshold, bool) or not np.isfinite(accuracy_threshold)
                    or not 0 <= accuracy_threshold <= 1):
        raise ValueError("reliability threshold must be a finite probability")
    if enabled and validation_accuracies is None:
        raise ValueError("Steering reliability requires pre-refit validation accuracies")
    candidates = range(1, len(vectors) - 1) if layers is None else layers
    active = []
    for layer in candidates:
        if not 0 <= layer < len(vectors):
            raise ValueError(f"Layer {layer} out of range for {len(vectors)} layers")
        if vectors[layer] is None:
            if layers is not None:
                raise FileNotFoundError(f"Missing concept vector for layer {layer}")
            continue
        if enabled:
            score = validation_accuracies[layer]
            if score is None or not np.isfinite(score):
                continue
            qualifies = (score >= accuracy_threshold if reliability_comparison == ">="
                         else score > accuracy_threshold)
            if not qualifies:
                continue
        active.append(int(layer))
    return active

def compute_adaptive_epsilon(
    embedding: torch.Tensor,
    concept_vector: torch.Tensor,
    bias: float,
    target_prob: float,
    *,
    probe_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute adaptive perturbation magnitude to steer towards target probability.
    
    For negative steering (steering away from positive/hate):
    - target_prob should be LOW (e.g., 0.0001) to make output negative/non-hate
    - Steers from positive (high P_m) towards negative (low P_m)
    - When current P_m is high (positive/hate), epsilon will be negative to steer away
    
    For positive steering (steering towards positive/hate):
    - target_prob should be HIGH (e.g., 0.99) to make output positive/hate
    - Steers from negative (low P_m) towards positive (high P_m)
    - When current P_m is low (negative/non-hate), epsilon will be positive to steer towards
    
    Formula: epsilon = I(should_steer) * (sigmoid^(-1)(P_target) - b - w^T*e)
    
    Args:
        embedding: Current embedding (batch_size, hidden_dim)
        concept_vector: Normalized concept vector (hidden_dim,) - this is w/||w||
        bias: Bias term for probability calculation
        target_prob: Target probability (e.g., 0.0001 for negative steering, 0.99 for positive steering)
    
    Returns:
        epsilon: Adaptive perturbation magnitude (negative to steer away, positive to steer towards)
    """
    # Optional calibrated raw-probe path. The legacy normalized classifier
    # below remains unchanged unless its own raw weight is explicitly supplied.
    if probe_weight is not None:
        if not 0.0 < target_prob < 1.0:
            raise ValueError("target_prob must lie strictly between zero and one")
        compute_dtype = torch.float32 if embedding.dtype in (torch.float16, torch.bfloat16) else embedding.dtype
        hidden = embedding.to(dtype=compute_dtype)
        weight = probe_weight.to(device=embedding.device, dtype=compute_dtype)
        direction = concept_vector.to(device=embedding.device, dtype=compute_dtype)
        if weight.shape != direction.shape or weight.shape[-1] != embedding.shape[-1]:
            raise ValueError("raw probe and injection direction must match the hidden dimension")
        slope = torch.sum(weight * direction)
        if not torch.isfinite(slope) or slope <= 0:
            raise ValueError("raw probe must have positive finite projection onto its own direction")
        current_logit = bias + torch.sum(hidden * weight, dim=-1, keepdim=True)
        target_logit = torch.log(torch.tensor(target_prob / (1 - target_prob), device=embedding.device, dtype=compute_dtype))
        current_prob = torch.sigmoid(current_logit)
        gate = current_prob < target_prob if target_prob > 0.5 else current_prob > target_prob
        return torch.where(gate, (target_logit - current_logit) / slope, torch.zeros_like(current_logit))

    # Convert target probability to logit: sigmoid^(-1)(P_target)
    logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob), device=embedding.device))
    
    # Since concept_vector is already L2 normalized, ||w|| = 1
    # Compute w^T*e (dot product)
    w_dot_e = torch.sum(embedding * concept_vector, dim=-1, keepdim=True)
    
    # Compute current logit: b + w^T*e
    current_logit = bias + w_dot_e
    current_prob = torch.sigmoid(current_logit)
    
    # Compute epsilon: (logit_target - current_logit)
    # This moves the embedding towards the target probability
    # For target_prob=0.0001 (negative): logit_target approx -9.21
    # If current_logit is large positive (P_m approx 1.0), epsilon will be large negative
    # Negative epsilon steers away from positive sentiment towards negative
    epsilon = logit_target - current_logit
    
    # Only apply if we haven't reached the target yet
    # For negative steering (target_prob < 0.5): steer when current_prob > target_prob (still too positive/non-hate)
    # For positive steering (target_prob > 0.5): steer when current_prob < target_prob (still too negative/non-hate)
    should_steer = (current_prob < target_prob if target_prob > 0.5 
                   else current_prob > target_prob).float()
    epsilon = epsilon * should_steer
    
    return epsilon

def register_steering_hooks_sequential(
    model,
    vectors: Sequence[Optional[torch.Tensor]],
    layer_ids: Sequence[int],
    *,
    target_prob: float = 0.0001,
    bias: float = 0.0,
    probe_weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
    probe_biases: Optional[Sequence[Optional[float]]] = None,
    validation_accuracies: Optional[Sequence[Optional[float]]] = None,
    test_accuracies: Optional[Sequence[Optional[float]]] = None,
    accuracy_threshold: float = 0.9,
    use_position_coefficients: bool = False,
    position_decay_rate: float = 0.1,
    verbose: bool = True,
    activation_site: str = "transformer_block_output",
    reliability_enabled: Optional[bool] = None,
    reliability_comparison: Optional[str] = ">",
):
    """
    Register adaptive steering hooks following sequential layer-by-layer steering algorithm.
    
    Adaptive Steering Algorithm:
        - Epsilon computed dynamically per layer to reach target_prob
        - Algorithm:
            for l = 1 to L do:
                if ValAcc(P_m) > P_1 then:
                    e <- Embedding at l-th layer after steering previous layers
                    if should_steer(e, P_target) then:
                        epsilon <- (sigmoid^(-1)(P_target) - b - w^T*e)  # Adaptive magnitude
                        e <- e + epsilon*v                        # Apply perturbation
                        - For target_prob < 0.5: negative epsilon steers away from positive/hate
                        - For target_prob > 0.5: positive epsilon steers towards positive/hate
    
    Position-based Steering (optional):
        - When use_position_coefficients=True, applies position-dependent weights
        - Uses exponential decay: weight(pos) = exp(-position_decay_rate * pos)
        - Earlier tokens receive stronger steering, later tokens receive weaker steering
    
    Args:
        target_prob: Target probability for adaptive steering (0.0001 = steer away from hate, 0.99 = steer towards hate)
        bias: Bias term for probability calculation (default 0.0)
        validation_accuracies: Pre-refit validation accuracies for the P_1 check
        accuracy_threshold: Threshold P_1 for pre-refit validation accuracy (default 0.9)
        reliability_enabled: Explicit task filtering policy; None retains legacy behavior.
        reliability_comparison: > or >= when enabled, None allowed when disabled.
        use_position_coefficients: Enable position-dependent steering strength (default False)
        position_decay_rate: Decay rate for exponential weighting (default 0.1)
        verbose: If True, print epsilon values for first token only (default True)
        activation_site: Explicit hook coordinate. The default preserves the
            original block-output path; the opt-in post_attention_layernorm
            output requires separately verified matching cache coordinates.
    """
    if (probe_weights is None) != (probe_biases is None):
        raise ValueError("raw calibration requires both own per-layer probe_weights and probe_biases")
    if test_accuracies is not None:
        raise ValueError("Steering reliability requires pre-refit validation accuracies; test accuracies cannot select layers")
    if activation_site not in {"transformer_block_output", "post_attention_layernorm"}:
        raise ValueError(f"Unsupported activation_site: {activation_site}")
    blocks = get_blocks(model)
    if activation_site == "post_attention_layernorm" and (
        not hasattr(model, "model") or not hasattr(model.model, "layers")
        or blocks is not model.model.layers
    ):
        raise ValueError("post_attention_layernorm requires the explicit model.layers block structure")
    handles = []
    
    # Sort layers to ensure sequential processing
    sorted_layers = sorted(layer_ids)
    active_layers = select_active_layers(
        vectors, validation_accuracies, layers=sorted_layers,
        accuracy_threshold=accuracy_threshold, reliability_enabled=reliability_enabled,
        reliability_comparison=reliability_comparison,
    )
    # Preflight every planned hook before registering any module. This also
    # prevents a missing later layer/probe from leaving an earlier intervention.
    planned_hooks = []
    for layer in active_layers:
        if not (0 <= layer < len(blocks)):
            raise ValueError(f"Layer {layer} out of range; model has {len(blocks)} layers.")
        vec = vectors[layer]
        if vec is None:
            raise FileNotFoundError(f"Missing concept vector for layer {layer}.")
        raw_weight = None if probe_weights is None else probe_weights[layer]
        layer_bias = bias if probe_biases is None else probe_biases[layer]
        if probe_weights is not None and (raw_weight is None or layer_bias is None):
            raise ValueError(f"Missing own raw probe weight/intercept for layer {layer}.")
        hook_module = (blocks[layer] if activation_site == "transformer_block_output"
                       else getattr(blocks[layer], "post_attention_layernorm", None))
        if hook_module is None or not callable(getattr(hook_module, "register_forward_hook", None)):
            raise ValueError(f"Layer {layer} has no supported {activation_site} hook module")
        planned_hooks.append((layer, hook_module, vec, raw_weight, layer_bias))

    # Flag to print only for first forward pass (first token generation)
    first_pass = {"done": False}

    print(f"\n[Adaptive Steering] Sequential layer-by-layer steering:")
    print(f"  - Target layers: {sorted_layers}")
    print(f"  - P_target (target_prob): {target_prob}")
    print(f"  - P_1 (accuracy_threshold): {accuracy_threshold}")
    print(f"  - Position-based steering: {use_position_coefficients}")
    if use_position_coefficients:
        print(f"  - Position decay rate: {position_decay_rate}")

    # Track generation step for position-based steering
    # This counter increments each time we generate a new token
    generation_step = {"count": 0}

    for layer, hook_module, vec, raw_weight, layer_bias in planned_hooks:
        def hook_fn(_module, _inputs, output, *, layer_idx=layer, base_vec=vec,
                    classifier_weight=raw_weight, classifier_bias=layer_bias,
                    last_layer=sorted_layers[-1], all_layers=sorted_layers):
            if isinstance(output, tuple):
                hidden, others = output[0], output[1:]
            else:
                hidden, others = output, None

            # Only steer the last token (the one being generated)
            hidden = hidden.clone()
            last = hidden[:, -1, :]
            steer_vec = base_vec.to(device=last.device, dtype=last.dtype)
            if steer_vec.shape[-1] != last.shape[-1]:
                raise ValueError(
                    f"Layer {layer_idx}: vector dim {steer_vec.shape[-1]} != hidden dim {last.shape[-1]}"
                )

            # Raw calibration retains this method's fitted weight and intercept.
            calibration_weight = (steer_vec if classifier_weight is None else
                                  classifier_weight.to(device=last.device, dtype=(
                                      torch.float32 if last.dtype in (torch.float16, torch.bfloat16) else last.dtype)))
            w_dot_e = torch.sum(last * calibration_weight, dim=-1, keepdim=True)
            current_logit = classifier_bias + w_dot_e
            current_prob = torch.sigmoid(current_logit)
            
            # Steer if current probability hasn't reached target yet
            # For target_prob > 0.5 (steer towards positive/hate): steer when current_prob < target_prob (still too low)
            # For target_prob < 0.5 (steer towards negative/non-hate): steer when current_prob > target_prob (still too high)
            should_steer = (current_prob.item() < target_prob if target_prob > 0.5 
                           else current_prob.item() > target_prob)
            
            # Initialize epsilon for DEBUG output
            epsilon = None
            
            if should_steer:
                # Compute adaptive epsilon to reach target_prob
                epsilon = compute_adaptive_epsilon(
                    embedding=last,
                    concept_vector=steer_vec,
                    bias=classifier_bias,
                    target_prob=target_prob,
                    probe_weight=calibration_weight if classifier_weight is not None else None,
                )
                
                # Apply position-based decay if enabled
                # Decay based on how many tokens have been generated so far
                if use_position_coefficients:
                    # Compute exponential decay with more moderate scaling
                    exp_weight = 1.5*torch.exp(torch.tensor(-position_decay_rate * generation_step["count"], 
                                                           device=epsilon.device, dtype=epsilon.dtype))
                    # Apply saturation: use max(exp_weight, 0.5) to allow some decay but maintain minimum steering
                    position_weight = torch.max(exp_weight, torch.tensor(1, device=epsilon.device, dtype=epsilon.dtype))
                    epsilon = epsilon * position_weight
                    
                    # Print with position weight info
                    if verbose and layer_idx == last_layer:
                        epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                        print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                              f"epsilon={epsilon_val:.4f}, weight={position_weight.item():.4f}, STEERING")
                else:
                    # Print epsilon values only for last layer (to avoid spam)
                    if verbose and layer_idx == last_layer:
                        epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                        print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                              f"epsilon={epsilon_val:.4f}, STEERING")
                
                # Apply perturbation: epsilon already has correct sign
                new_last = last + epsilon * steer_vec
                
                # DEBUG: Print steering effect for last layer
                if verbose and layer_idx == last_layer:
                    new_w_dot_e = torch.sum(new_last * calibration_weight, dim=-1, keepdim=True)
                    new_logit = classifier_bias + new_w_dot_e
                    new_prob = torch.sigmoid(new_logit)
                    logit_change = (new_logit - current_logit).item()
                    print(f"    -> After steering: P_m={new_prob.item():.4f}, logit={new_logit.item():.2f} "
                          f"(change={logit_change:.2f}), w_dot_e={new_w_dot_e.item():.2f}")
            else:
                # Already at target probability, no steering needed
                if verbose and layer_idx == last_layer:
                    print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                          f"SKIP (already at target)")
                new_last = last
            
            hidden[:, -1, :] = new_last
            
            # DEBUG: Print info for all layers in Step 0 to verify steering is applied
            if verbose and generation_step["count"] == 0:  # Only for first token to avoid spam
                final_w_dot_e = torch.sum(new_last * calibration_weight, dim=-1, keepdim=True)
                final_logit = classifier_bias + final_w_dot_e
                final_prob = torch.sigmoid(final_logit)
                # Get epsilon value (already computed if should_steer is True)
                if epsilon is not None:
                    epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                else:
                    epsilon_val = 0.0
                print(f"  [DEBUG Step 0][Layer {layer_idx}] Final P_m={final_prob.item():.4f}, "
                      f"epsilon={epsilon_val:.2f}, steered={should_steer}")
            
            # Increment generation step counter after last layer processes
            # This tracks how many tokens have been generated
            if layer_idx == last_layer:
                generation_step["count"] += 1

            return (hidden, *others) if others is not None else hidden

        try:
            handles.append(hook_module.register_forward_hook(hook_fn))
        except BaseException:
            for handle in handles:
                handle.remove()
            raise

    return handles
