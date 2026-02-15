import logging
import os
from pathlib import Path
from pprint import pformat

import equinox as eqx
import grain.python as grain
import hydra
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from optax import OptState
from tqdm import tqdm as _tqdm

from ttt.config import Config, register_configs
from ttt.dataloader.lm_dataset import dummy_dataset, lm_dataset
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.infra.wandb_utils import WandbLogger
from ttt.model.loop import Evaluator, train_on_sequence
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel
from ttt.optimizers import make_optimizer
from ttt.utils.jax_utils import eval_shape_and_sharding, get_custom_tqdm, initialize_distibuted, master_log, set_random_seed, tree_rearrange

register_configs()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _prepare_data_parallelism(cfg: Config, global_dev_num: int) -> int:
    if cfg.training.n_data_parallel is None:
        assert global_dev_num % cfg.training.n_state_parallel == 0, "Number of devices must be divisible by state parallelism"
        cfg.training.n_data_parallel = global_dev_num // cfg.training.n_state_parallel
    assert cfg.training.n_data_parallel * cfg.training.n_state_parallel == global_dev_num, (
        f"Data parallelism ({cfg.training.n_data_parallel}) and state parallelism ({cfg.training.n_state_parallel}) must match the number of devices ({global_dev_num})"
    )
    return cfg.training.n_data_parallel


def _make_train_iterator(cfg: Config, model_cfg, data_sharding: jax.sharding.Sharding, n_data_parallel: int):
    train_ds = (
        lm_dataset(
            path=cfg.training.dataset_path,
            split=cfg.training.data_split,
            seq_len=cfg.training.seq_length,
            seed=cfg.training.data_seed,
            global_batch_size=cfg.training.global_batch_size,
            repeat=True,
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
        )
        if not cfg.training.dummy_dataset
        else dummy_dataset(
            seq_len=cfg.training.seq_length,
            global_batch_size=cfg.training.global_batch_size,
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
            repeat=True,
        )
    )

    def load_to_sharded_array(arr):
        return jax.make_array_from_process_local_data(sharding=data_sharding, local_data=arr, global_shape=(cfg.training.global_batch_size, *arr.shape[1:]))

    def to_sharded_batch(batch):
        batch = jax.tree.map(lambda x: load_to_sharded_array(x), batch)
        return tree_rearrange(batch, "(data_parallel batch) ... -> data_parallel batch ...", data_parallel=n_data_parallel)

    train_iter_ds = train_ds.to_iter_dataset(
        grain.ReadOptions(num_threads=cfg.training.loader_workers, prefetch_buffer_size=500),
    )
    return iter(train_iter_ds), to_sharded_batch


def _main(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.info("\n".join([f"{k}={v}" for k, v in os.environ.items()]))
    logger.info(f"Launching with \n {pformat(cfg_dict)}.")

    model_cfg = cfg.model
    backend_cfg = cfg.backend

    initialize_distibuted(backend_cfg)

    key = set_random_seed(cfg.training.model_seed)

    n_host = jax.process_count()

    global_dev_num = jax.device_count()
    local_dev_num = jax.local_device_count()
    master_process = jax.process_index() == 0

    n_data_parallel = _prepare_data_parallelism(cfg, global_dev_num)

    log_dir = Path(cfg.training.exp_dir) / cfg.training.exp_folder / cfg.training.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        entity=cfg.training.wandb_entity,
        project=cfg.training.wandb_project,
        exp_name=cfg.training.exp_name,
        load_part=cfg.training.load_part,
        log_dir=log_dir,
        wandb_key=cfg.training.wandb_key,
        logging_process=0,
        config=cfg_dict,
        enabled=cfg.training.log_wandb,
    )

    dev_info = f"Process # {n_host}\tLocal dev # {local_dev_num}\tTotal dev # {global_dev_num}"
    master_log(logger, dev_info)

    checkpointer = Checkpointer(config=cfg, for_saving=True)

    optimizer_outer_loop, optimizer_info_outer_loop = make_optimizer(cfg.training.optimizer_outer)

    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh
    data_sharding = jax.NamedSharding(mesh, P("data"))
    cfg.model.seq_len = cfg.training.seq_length

    train_ds_iter, to_sharded_batch = _make_train_iterator(cfg, model_cfg, data_sharding, n_data_parallel)

    @eqx.filter_jit
    def create_sharded_model_and_state() -> tuple[MetaModel, eqx.nn.State]:
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))  # Replicate initial (empty) state
        model = model_sharding.shard_params(model)
        return model, state

    @eqx.filter_jit
    def create_stepped_opt_state(model: MetaModel) -> OptState:
        """
        Create optimizer state with correct sharding after having a single update step applied.
        """
        trainable_params = model.trainable_parameters()
        opt_state = optimizer_outer_loop.init(trainable_params)
        _, opt_state = optimizer_outer_loop.update(trainable_params, opt_state, model.trainable_parameters())
        # Should be sharded the same way as the model parameters
        return opt_state

    continued_run = wandb_logger.preexisting
    master_log(logger, f"Wandb preexisting: {continued_run}")
    if (continued_run and checkpointer.checkpoint_exists()) or cfg.training.load_part != "none":
        if continued_run and checkpointer.checkpoint_exists():
            load_part = "all"  # Resuming from the current checkpointing directory requires the optimizer and loop state
            load_checkpointer = checkpointer
        else:
            assert cfg.checkpoint.resume_checkpoint_dir is not None
            load_part = cfg.training.load_part
            load_checkpointer = Checkpointer(
                config=cfg, for_saving=False
            )  # Use the resumption path only if the run is starting from scratch. Otherwise use the current checkpointing path.

        if load_part == "all" and cfg.training.eval_mode:  # prevent uncessary opt and loop state resumption
            load_part = "params"

        abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
        abstract_opt_state = eval_shape_and_sharding(lambda: create_stepped_opt_state(create_sharded_model_and_state()[0]))

        out_state = load_checkpointer.load_checkpoint(
            step=cfg.training.resume_step,
            targets={"model_weights": abstract_model_weights, "opt_state": abstract_opt_state, "train_ds_iter": train_ds_iter},
            restore=load_part,
        )

        def load_model_weights(model: MetaModel, out_state) -> MetaModel:
            model_loaded = unify_dict_with_eqx_module(out_state["model_weights"], model)[0]
            return model_loaded

        master_log(logger, "Restoring model weights")
        model, state = create_sharded_model_and_state()
        model = load_model_weights(model, out_state)

        if "opt_state" not in out_state:  # Create new optimizer state
            master_log(logger, "Restored model weights, creating new optimizer state")
            opt_state = optimizer_outer_loop.init(model.trainable_parameters())
            start_step = 0

        else:  # Restore optimizer state

            def create_opt_state_with_loaded_weights(model: MetaModel, out_state) -> OptState:
                opt_state = create_stepped_opt_state(model)
                opt_state = unify_dict_with_eqx_module(out_state["opt_state"], opt_state)[0]
                return opt_state

            master_log(logger, "Restoring optimizer state")
            opt_state = create_opt_state_with_loaded_weights(model, out_state)
            start_step = int(jax.device_get(out_state["train_ds_iter"].get_state()["next_index"]))

        del out_state, load_checkpointer

    else:  # Create new model and optimizer state
        model, state = create_sharded_model_and_state()
        opt_state = optimizer_outer_loop.init(model.trainable_parameters())  # Sharding taken from model
        start_step = 0

    ### Include Storage
    num_trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(model.trainable_parameters()))
    num_non_embedding_params = num_trainable_params - model.language_model.model.wte.weight.size
    if model.language_model.lm_head is not None:
        num_non_embedding_params -= model.language_model.lm_head.weight.size
    logger.info(f"#Trainable params: {num_trainable_params}")
    logger.info(f"#Non-embed params: {num_non_embedding_params}")

    M = MetaModel.MetricType
    evaluator = Evaluator(
        global_batch_size=max(cfg.training.eval_batch_size, cfg.training.global_batch_size // cfg.training.accum_steps * 4),  # Larger bs to speed up eval
        data_sharding=data_sharding,
        config=cfg,
        wandb_logger=wandb_logger,
        log_dir=log_dir,
    )

    total_steps = cfg.training.total_steps
    assert total_steps >= 1, "Total step must >=1, otherwise, lower global batch size"
    master_log(logger, f"Total steps: {total_steps}")

    with mesh:
        if cfg.training.eval_mode or start_step == total_steps:
            state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
            evaluator.eval_fn(model, state, start_step)
            jax.experimental.multihost_utils.sync_global_devices("eval finished")
            return

        tqdm = get_custom_tqdm() if master_process else _tqdm
        for step in tqdm(range(start_step, total_steps), initial=start_step, total=total_steps, desc="Outer Loop Training", disable=not master_process):
            if 0 < cfg.training.break_step < step:
                jax.experimental.multihost_utils.sync_global_devices("reached break step")
                break

            batch = to_sharded_batch(next(train_ds_iter))

            state = state.set(model.step_index, jnp.array(step, dtype=jnp.int32))
            model, opt_state, loss, metrics = train_on_sequence(state, model, opt_state, batch, cfg)
            loss_ce = metrics[M.loss].mean()

            update = {
                "loss": jax.device_get(loss_ce).item(),
                "gradient_norm": jax.device_get(metrics[M.outer_grad_norm]).item(),
                "outer_learning_rate": jnp.asarray(optimizer_info_outer_loop["learning_rate_schedule"](int(opt_state[1][2].count) - 1)).item(),
            }

            wandb_logger.log(update, step)

            if (cfg.training.save_milestone_freq > 0 and step % cfg.training.save_milestone_freq == 0 and step != 0) or (step == cfg.training.total_steps - 1):
                master_log(logger, f"Saving checkpoint at step {step}, do not kill...")
                is_milestone = (cfg.training.save_milestone_freq > 0) and (step % cfg.training.save_milestone_freq == 0)

                checkpointer.save_checkpoint(
                    step=step,
                    model=model,
                    opt_state=opt_state,
                    train_ds_iter=train_ds_iter,
                    is_milestone=is_milestone,
                )

                # Make sure the previous checkpoint is finished since we'll donate the weights the next loop
                checkpointer.wait_until_finished()

                if step == cfg.training.total_steps - 1:
                    evaluator.eval_fn(model, state, step)

        checkpointer.close()  # Always wait until checkpoints are done saving

        if cfg.backend.distributed:
            jax.experimental.multihost_utils.sync_global_devices("end_of_training")


@hydra.main(version_base=None, config_path=str(Path("configs").absolute().resolve()), config_name="config")
def main(cfg: Config):
    if cfg.backend.compilation_cache_dir is not None:
        import jax

        jax.config.update("jax_compilation_cache_dir", cfg.backend.compilation_cache_dir)

    _main(cfg)


if __name__ == "__main__":
    main()
