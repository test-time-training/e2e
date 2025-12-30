import jax
import jax.numpy as jnp
import pytest
from ttt.optimization.adaptive_lr import (
    AdaptiveLRConfig,
    AdaptiveLROptimizer,
)


class TestAdaptiveLROptimizer:
    """Test suite for adaptive learning rate optimizer."""
    
    @pytest.fixture
    def config(self):
        return AdaptiveLRConfig(
            enabled=True,
            base_lr=0.01,
            ema_decay=0.99,
            grad_norm_decay=0.95,
            min_lr=1e-5,
            max_lr=0.1,
        )
    
    @pytest.fixture
    def optimizer(self, config):
        return AdaptiveLROptimizer(config)
    
    @pytest.fixture
    def dummy_params(self):
        """Create dummy transformer-like params."""
        return {
            "attention/query": jnp.ones((768, 768)),
            "attention/key": jnp.ones((768, 768)),
            "mlp/dense1": jnp.ones((768, 3072)),
            "mlp/dense2": jnp.ones((3072, 768)),
        }
    
    @pytest.fixture
    def dummy_grads(self, dummy_params):
        """Create dummy gradients."""
        return jax.tree_map(
            lambda x: 0.001 * jnp.ones_like(x),
            dummy_params
        )
    
    def test_initialization(self, optimizer, dummy_params):
        """Test optimizer initializes correctly."""
        state = optimizer.init(dummy_params)
        
        assert state.step == 0
        assert state.loss_ema == 0.0
        assert len(state.layer_lrs) == 4
        assert len(state.grad_norm_ema) == 4
        assert len(state.momentum_states) == 4
        
        # All initial LRs should be base_lr
        for lr in state.layer_lrs.values():
            assert float(lr) == pytest.approx(0.01)
    
    def test_single_update_step(self, optimizer, dummy_params, dummy_grads):
        """Test single update step."""
        state = optimizer.init(dummy_params)
        loss = jnp.array(2.5)
        
        new_params, new_state, metrics = optimizer.update(
            state, dummy_grads, dummy_params, loss
        )
        
        # Check state was updated
        assert new_state.step == 1
        assert new_state.loss_ema == pytest.approx(float(loss))
        
        # Check metrics
        assert "adaptive_lr/avg_lr" in metrics
        assert "adaptive_lr/loss" in metrics
        assert "adaptive_lr/step" in metrics
        assert metrics["adaptive_lr/step"] == 1
    
    def test_loss_ema_accumulation(self, optimizer, dummy_params, dummy_grads):
        """Test that loss EMA accumulates correctly."""
        state = optimizer.init(dummy_params)
        
        losses = [2.5, 2.3, 2.1, 2.0, 1.9]
        for loss_val in losses:
            loss = jnp.array(loss_val)
            _, state, _ = optimizer.update(
                state, dummy_grads, dummy_params, loss
            )
        
        # Loss EMA should be closer to recent values than initial
        assert float(state.loss_ema) > 1.9
        assert float(state.loss_ema) < 2.5
    
    def test_gradient_norm_tracking(self, optimizer, dummy_params):
        """Test gradient norm statistics."""
        state = optimizer.init(dummy_params)
        
        # Step 1: small gradients
        small_grads = jax.tree_map(
            lambda x: 1e-4 * jnp.ones_like(x),
            dummy_params
        )
        _, state, metrics1 = optimizer.update(
            state, small_grads, dummy_params, jnp.array(2.0)
        )
        
        grad_norm_1 = metrics1["adaptive_lr/avg_grad_norm"]
        
        # Step 2: large gradients (should trigger LR increase)
        large_grads = jax.tree_map(
            lambda x: 0.1 * jnp.ones_like(x),
            dummy_params
        )
        _, state, metrics2 = optimizer.update(
            state, large_grads, dummy_params, jnp.array(2.0)
        )
        
        grad_norm_2 = metrics2["adaptive_lr/avg_grad_norm"]
        
        # Gradient norms should differ
        assert grad_norm_1 != grad_norm_2
    
    def test_learning_rate_bounds(self, optimizer, dummy_params):
        """Test that learning rates stay within bounds."""
        state = optimizer.init(dummy_params)
        
        # Run many steps with various gradient magnitudes
        for _ in range(100):
            grads = jax.tree_map(
                lambda x: jnp.random.normal(jnp.shape(x)) * 0.01,
                dummy_params,
                key=jax.random.PRNGKey(0),
            )
            _, state, _ = optimizer.update(
                state, grads, dummy_params, jnp.array(1.5)
            )
        
        # All LRs should be within bounds
        for lr in state.layer_lrs.values():
            assert float(lr) >= 1e-5
            assert float(lr) <= 0.1
    
    def test_momentum_accumulation(self, optimizer, dummy_params):
        """Test momentum state accumulation."""
        state = optimizer.init(dummy_params)
        
        # All initial momentum states should be zero
        for mom in state.momentum_states.values():
            assert jnp.allclose(mom, 0.0)
        
        # After updates, momentum states should be non-zero
        grads = jax.tree_map(
            lambda x: 0.01 * jnp.ones_like(x),
            dummy_params
        )
        _, new_state, _ = optimizer.update(
            state, grads, dummy_params, jnp.array(2.0)
        )
        
        # Check that momentum accumulated
        for old_mom, new_mom in zip(
            state.momentum_states.values(),
            new_state.momentum_states.values()
        ):
            assert not jnp.allclose(new_mom, 0.0)
    
    def test_param_update_direction(self, optimizer, dummy_params):
        """Test that parameters move in correct direction."""
        state = optimizer.init(dummy_params)
        
        # Positive gradients should decrease parameters
        grads = jax.tree_map(
            lambda x: 0.01 * jnp.ones_like(x),
            dummy_params
        )
        
        new_params, _, _ = optimizer.update(
            state, grads, dummy_params, jnp.array(2.0)
        )
        
        # Check parameter updates
        for key in dummy_params.keys():
            assert jnp.all(new_params[key] < dummy_params[key])
    
    def test_saturation_detection(self, optimizer, dummy_params):
        """Test saturation detection and LR adjustment."""
        state = optimizer.init(dummy_params)
        
        # Initialize learning rates
        initial_lrs = {k: float(v) for k, v in state.layer_lrs.items()}
        
        # Simulate saturation: small gradients for many steps
        for _ in range(10):
            small_grads = jax.tree_map(
                lambda x: 1e-6 * jnp.ones_like(x),
                dummy_params
            )
            _, state, _ = optimizer.update(
                state, small_grads, dummy_params, jnp.array(1.5)
            )
        
        # LRs should decrease due to saturation
        final_lrs = {k: float(v) for k, v in state.layer_lrs.items()}
        
        for key in initial_lrs.keys():
            assert final_lrs[key] < initial_lrs[key]
