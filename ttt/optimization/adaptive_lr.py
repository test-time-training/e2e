"""
Adaptive Learning Rate Scheduling for Test-Time Training
This module implements dynamic learning rate adjustment during test-time training
based on gradient statistics and loss trajectory.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Dict, Tuple, Optional
import optax


class AdaptiveLRState(NamedTuple):
    """State for adaptive learning rate scheduling."""
    step: int
    loss_ema: jnp.ndarray  # Exponential moving average of loss
    grad_norm_ema: jnp.ndarray  # EMA of gradient norms per layer
    layer_lrs: Dict[str, jnp.ndarray]  # Current LR for each layer
    momentum_states: Dict[str, jnp.ndarray]  # Momentum for each layer


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler for test-time training.
    
    Adjusts layer-wise learning rates based on:
    - Loss trajectory (EMA)
    - Gradient norm statistics
    - Learning saturation detection
    """
    
    def __init__(
        self,
        base_lr: float = 0.01,
        ema_decay: float = 0.99,
        grad_norm_decay: float = 0.95,
        min_lr: float = 1e-5,
        max_lr: float = 0.1,
        saturation_threshold: float = 0.95,
        scale_factor: float = 1.5,
    ):
        """
        Initialize adaptive LR scheduler.
        
        Args:
            base_lr: Base learning rate for all layers
            ema_decay: Decay for loss EMA (higher = more smoothing)
            grad_norm_decay: Decay for gradient norm EMA
            min_lr: Minimum learning rate per layer
            max_lr: Maximum learning rate per layer
            saturation_threshold: Ratio of recent to old loss for saturation detection
            scale_factor: Factor to scale LR changes
        """
        self.base_lr = base_lr
        self.ema_decay = ema_decay
        self.grad_norm_decay = grad_norm_decay
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.saturation_threshold = saturation_threshold
        self.scale_factor = scale_factor
    
    def init(self, params: Dict) -> AdaptiveLRState:
        """Initialize scheduler state from model parameters."""
        layer_lrs = jax.tree_map(
            lambda x: jnp.full((1,), self.base_lr),
            params
        )
        grad_norm_ema = jax.tree_map(
            lambda x: jnp.array(0.0),
            params
        )
        momentum_states = jax.tree_map(
            lambda x: jnp.zeros_like(x),
            params
        )
        
        return AdaptiveLRState(
            step=0,
            loss_ema=jnp.array(0.0),
            grad_norm_ema=grad_norm_ema,
            layer_lrs=layer_lrs,
            momentum_states=momentum_states,
        )
    
    def update(
        self,
        state: AdaptiveLRState,
        loss: jnp.ndarray,
        grads: Dict,
        params: Dict,
    ) -> Tuple[AdaptiveLRState, Dict[str, float]]:
        """
        Update learning rates and return updated state with metrics.
        
        Args:
            state: Current scheduler state
            loss: Current loss value
            grads: Gradients from backprop
            params: Current model parameters
            
        Returns:
            Updated state and dictionary of metrics
        """
        step = state.step + 1
        
        # Update loss EMA
        loss_ema = jnp.where(
            state.step == 0,
            loss,
            self.ema_decay * state.loss_ema + (1 - self.ema_decay) * loss
        )
        
        # Compute per-layer gradient norms
        def compute_grad_norm(g):
            return jnp.sqrt(jnp.sum(g ** 2))
        
        grad_norms = jax.tree_map(compute_grad_norm, grads)
        
        # Update gradient norm EMAs
        def update_grad_ema(old_ema, new_norm):
            return jnp.where(
                state.step == 0,
                new_norm,
                self.grad_norm_decay * old_ema + (1 - self.grad_norm_decay) * new_norm
            )
        
        new_grad_norm_ema = jax.tree_map(
            update_grad_ema,
            state.grad_norm_ema,
            grad_norms
        )
        
        # Compute layer-wise learning rate adjustments
        def adjust_lr(layer_name, old_lr, grad_norm, grad_norm_ema):
            # Detection: gradient is decreasing (saturation)
            saturation_ratio = grad_norm / (grad_norm_ema + 1e-8)
            is_saturating = saturation_ratio < self.saturation_threshold
            
            # Decrease LR if saturating, increase if actively learning
            lr_scale = jnp.where(
                is_saturating,
                1.0 / self.scale_factor,  # Decrease
                self.scale_factor,  # Increase
            )
            
            new_lr = old_lr[0] * lr_scale
            new_lr = jnp.clip(new_lr, self.min_lr, self.max_lr)
            
            return jnp.array([new_lr])
        
        # Tree map to update all layer LRs
        new_layer_lrs = {}
        for key in state.layer_lrs.keys():
            old_lr = state.layer_lrs[key]
            grad_norm = grad_norms[key]
            grad_norm_ema = new_grad_norm_ema[key]
            new_layer_lrs[key] = adjust_lr(
                key, old_lr, grad_norm, grad_norm_ema
            )
        
        # Compute metrics
        avg_lr = jnp.mean(jnp.array([lr[0] for lr in new_layer_lrs.values()]))
        avg_grad_norm = jnp.mean(jnp.array(list(grad_norms.values())))
        
        metrics = {
            "loss": float(loss),
            "loss_ema": float(loss_ema),
            "avg_lr": float(avg_lr),
            "avg_grad_norm": float(avg_grad_norm),
            "step": int(step),
        }
        
        new_state = AdaptiveLRState(
            step=step,
            loss_ema=loss_ema,
            grad_norm_ema=new_grad_norm_ema,
            layer_lrs=new_layer_lrs,
            momentum_states=state.momentum_states,
        )
        
        return new_state, metrics


class AdaptiveOptimizer:
    """
    Wraps AdaptiveLRScheduler with parameter updates.
    Implements adaptive learning rate + momentum.
    """
    
    def __init__(
        self,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        use_nesterov: bool = True,
        **scheduler_kwargs
    ):
        self.scheduler = AdaptiveLRScheduler(base_lr=base_lr, **scheduler_kwargs)
        self.momentum = momentum
        self.use_nesterov = use_nesterov
    
    def init(self, params: Dict) -> AdaptiveLRState:
        return self.scheduler.init(params)
    
    def update(
        self,
        state: AdaptiveLRState,
        grads: Dict,
        params: Dict,
        loss: jnp.ndarray,
    ) -> Tuple[Dict, AdaptiveLRState, Dict]:
        """
        Apply adaptive learning rate update to parameters.
        
        Returns:
            Updated parameters, updated state, and metrics
        """
        # Update learning rates
        state, metrics = self.scheduler.update(state, loss, grads, params)
        
        # Apply updates with momentum
        def update_param(param, grad, old_momentum, lr):
            # Momentum update
            new_momentum = self.momentum * old_momentum + grad
            
            if self.use_nesterov:
                update = self.momentum * new_momentum + grad
            else:
                update = new_momentum
            
            # Apply learning rate (which is per-layer)
            new_param = param - lr[0] * update
            return new_param, new_momentum
        
        new_params = {}
        new_momentum_states = {}
        
        for key in params.keys():
            param = params[key]
            grad = grads[key]
            old_momentum = state.momentum_states[key]
            lr = state.layer_lrs[key]
            
            new_param, new_momentum = update_param(param, grad, old_momentum, lr)
            new_params[key] = new_param
            new_momentum_states[key] = new_momentum
        
        # Update state with new momentum
        new_state = AdaptiveLRState(
            step=state.step,
            loss_ema=state.loss_ema,
            grad_norm_ema=state.grad_norm_ema,
            layer_lrs=state.layer_lrs,
            momentum_states=new_momentum_states,
        )
        
        return new_params, new_state, metrics


def create_adaptive_optax_schedule(
    base_lr: float = 0.01,
    num_steps: int = 1000,
    warmup_steps: int = 100,
) -> optax.Schedule:
    """
    Create an optax learning rate schedule that pairs with adaptive scheduler.
    This provides coarse-grained scheduling while adaptive handles fine-grained adjustment.
    """
    def schedule(step):
        # Warmup phase
        warmup_lr = base_lr * jnp.minimum(step / warmup_steps, 1.0)
        
        # Cosine annealing after warmup
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine_lr = base_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        
        # Use warmup first, then cosine
        lr = jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr
    
    return schedule


# Example usage in training loop
def example_training_step(
    params: Dict,
    state: AdaptiveLRState,
    optimizer: AdaptiveOptimizer,
    loss_fn,
    input_batch,
):
    """Example of how to use AdaptiveOptimizer in a training loop."""
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, input_batch)
    
    # Update with adaptive learning rates
    new_params, new_state, metrics = optimizer.update(
        state, grads, params, loss
    )
    
    return new_params, new_state, metrics


if __name__ == "__main__":
    # Minimal test
    print("Adaptive LR Scheduler initialized successfully")
    
    # Create dummy parameters
    dummy_params = {
        "layer1": jnp.ones((10, 10)),
        "layer2": jnp.ones((10, 5)),
    }
    
    optimizer = AdaptiveOptimizer(base_lr=0.01)
    state = optimizer.init(dummy_params)
    
    # Simulate a step
    dummy_grads = jax.tree_map(lambda x: 0.01 * jnp.ones_like(x), dummy_params)
    dummy_loss = jnp.array(1.5)
    
    new_params, new_state, metrics = optimizer.update(
        state, dummy_grads, dummy_params, dummy_loss
    )
    
    print(f"Step {metrics['step']}: Loss={metrics['loss']:.4f}, Avg LR={metrics['avg_lr']:.6f}")       
"""
Integration module for Adaptive Learning Rate Scheduling in E2E TTT framework.
Drop this into ttt/optimization/adaptive_lr.py
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate scheduling."""
    enabled: bool = True
    base_lr: float = 0.01
    ema_decay: float = 0.99
    grad_norm_decay: float = 0.95
    min_lr: float = 1e-5
    max_lr: float = 0.1
    saturation_threshold: float = 0.95
    scale_factor: float = 1.5
    momentum: float = 0.9
    use_nesterov: bool = True
    log_per_layer: bool = False


@struct.dataclass
class AdaptiveLRState:
    """Flax-compatible state for adaptive LR scheduling."""
    step: jnp.ndarray
    loss_ema: jnp.ndarray
    grad_norm_ema: Dict[str, jnp.ndarray]
    layer_lrs: Dict[str, jnp.ndarray]
    momentum_states: Dict[str, jnp.ndarray]


class AdaptiveLROptimizer:
    """
    Integrates with E2E training loop for per-layer adaptive learning rates.
    Works with standard JAX optax optimizers as a wrapper.
    """
    
    def __init__(self, config: AdaptiveLRConfig):
        self.config = config
    
    def init(self, params: Dict[str, Any]) -> AdaptiveLRState:
        """Initialize optimizer state from model params."""
        
        # Flatten param tree to get layer names
        flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
        
        layer_lrs = {}
        grad_norm_ema = {}
        momentum_states = {}
        
        for key_path, param in flat_params:
            # Convert key path to string
            key_str = "/".join(str(k.key) for k in key_path)
            
            layer_lrs[key_str] = jnp.array(self.config.base_lr)
            grad_norm_ema[key_str] = jnp.array(0.0)
            momentum_states[key_str] = jnp.zeros_like(param)
        
        return AdaptiveLRState(
            step=jnp.array(0, dtype=jnp.int32),
            loss_ema=jnp.array(0.0),
            grad_norm_ema=grad_norm_ema,
            layer_lrs=layer_lrs,
            momentum_states=momentum_states,
        )
    
    def update(
        self,
        state: AdaptiveLRState,
        grads: Dict[str, Any],
        params: Dict[str, Any],
        loss: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], AdaptiveLRState, Dict[str, float]]:
        """
        Update parameters with adaptive per-layer learning rates.
        
        Args:
            state: Current optimizer state
            grads: Gradients from backprop
            params: Current model parameters
            loss: Current loss value
            
        Returns:
            (updated_params, updated_state, metrics_dict)
        """
        
        step = state.step + 1
        
        # Update loss EMA
        loss_ema = jnp.where(
            state.step == 0,
            loss,
            self.config.ema_decay * state.loss_ema + 
            (1 - self.config.ema_decay) * loss
        )
        
        # Flatten trees to iterate
        flat_grads, grad_tree_def = jax.tree_util.tree_flatten(grads)
        flat_params, param_tree_def = jax.tree_util.tree_flatten(params)
        
        # Compute per-layer gradient norms
        grad_norms = [
            jnp.sqrt(jnp.sum(g ** 2)) for g in flat_grads
        ]
        
        # Update gradient norm EMAs and compute new LRs
        new_layer_lrs = {}
        new_grad_norm_ema = {}
        layer_metrics = {}
        
        for idx, (layer_name, grad_norm) in enumerate(
            zip(state.layer_lrs.keys(), grad_norms)
        ):
            old_lr = state.layer_lrs[layer_name]
            old_ema = state.grad_norm_ema[layer_name]
            
            # Update EMA of gradient norm
            new_ema = jnp.where(
                state.step == 0,
                grad_norm,
                self.config.grad_norm_decay * old_ema + 
                (1 - self.config.grad_norm_decay) * grad_norm
            )
            new_grad_norm_ema[layer_name] = new_ema
            
            # Detect saturation: compare current to recent history
            saturation_ratio = grad_norm / (old_ema + 1e-8)
            is_saturating = saturation_ratio < self.config.saturation_threshold
            
            # Adjust learning rate
            lr_scale = jnp.where(
                is_saturating,
                1.0 / self.config.scale_factor,
                self.config.scale_factor,
            )
            
            new_lr = jnp.clip(
                old_lr * lr_scale,
                self.config.min_lr,
                self.config.max_lr,
            )
            new_layer_lrs[layer_name] = new_lr
            
            if self.config.log_per_layer:
                layer_metrics[layer_name] = {
                    "lr": float(new_lr),
                    "grad_norm": float(grad_norm),
                    "saturation_ratio": float(saturation_ratio),
                }
        
        # Apply parameter updates with momentum
        new_params = {}
        new_momentum_states = {}
        
        for idx, (param, grad) in enumerate(zip(flat_params, flat_grads)):
            layer_name = list(state.layer_lrs.keys())[idx]
            lr = new_layer_lrs[layer_name]
            old_momentum = state.momentum_states[layer_name]
            
            # Momentum update
            new_momentum = (
                self.config.momentum * old_momentum + grad
            )
            
            if self.config.use_nesterov:
                update = self.config.momentum * new_momentum + grad
            else:
                update = new_momentum
            
            # Parameter update
            new_param = param - lr * update
            new_params[layer_name] = new_param
            new_momentum_states[layer_name] = new_momentum
        
        # Reconstruct param tree
        new_params_tree = jax.tree_util.tree_unflatten(
            param_tree_def, list(new_params.values())
        )
        
        # Create metrics
        avg_lr = jnp.mean(jnp.array(list(new_layer_lrs.values())))
        avg_grad_norm = jnp.mean(jnp.array(grad_norms))
        
        metrics = {
            "adaptive_lr/avg_lr": float(avg_lr),
            "adaptive_lr/loss": float(loss),
            "adaptive_lr/loss_ema": float(loss_ema),
            "adaptive_lr/avg_grad_norm": float(avg_grad_norm),
            "adaptive_lr/step": int(step),
        }
        
        if self.config.log_per_layer:
            metrics["adaptive_lr/per_layer"] = layer_metrics
        
        # Update state
        new_state = AdaptiveLRState(
            step=step,
            loss_ema=loss_ema,
            grad_norm_ema=new_grad_norm_ema,
            layer_lrs=new_layer_lrs,
            momentum_states=new_momentum_states,
        )
        
        return new_params_tree, new_state, metrics


def create_adaptive_lr_optimizer(config: AdaptiveLRConfig) -> AdaptiveLROptimizer:
    """Factory function to create adaptive LR optimizer."""
    return AdaptiveLROptimizer(config)


# Integration with Hydra config (add to configs/optimizer/adaptive_lr.yaml)
HYDRA_CONFIG_TEMPLATE = """
# configs/optimizer/adaptive_lr.yaml
optimizer:
  _target_: ttt.optimization.adaptive_lr.create_adaptive_lr_optimizer
  config:
    enabled: true
    base_lr: 0.01
    ema_decay: 0.99
    grad_norm_decay: 0.95
    min_lr: 1e-5
    max_lr: 0.1
    saturation_threshold: 0.95
    scale_factor: 1.5
    momentum: 0.9
    use_nesterov: true
    log_per_layer: false
"""


# Example integration into E2E trainer
TRAINER_INTEGRATION_TEMPLATE = """
# In ttt/train.py or your training module

from ttt.optimization.adaptive_lr import create_adaptive_lr_optimizer, AdaptiveLRConfig

def train_step(
    state,  # TrainState with params, opt_state, etc.
    batch,
    model,
    loss_fn,
    adaptive_lr_optimizer,  # NEW
):
    '''Single training step with adaptive learning rates.'''
    
    def loss_and_grads_fn(params):
        logits = model.apply({'params': params}, batch['input_ids'])
        loss = loss_fn(logits, batch['labels'])
        return loss, jax.grad(loss_fn)(params, batch['input_ids'], batch['labels'])
    
    loss, grads = loss_and_grads_fn(state.params)
    
    # NEW: Update with adaptive learning rates
    new_params, adaptive_state, metrics = adaptive_lr_optimizer.update(
        state.adaptive_lr_state,
        grads,
        state.params,
        loss,
    )
    
    # Update main training state
    new_state = state.replace(
        params=new_params,
        adaptive_lr_state=adaptive_state,
    )
    
    return new_state, metrics
"""
