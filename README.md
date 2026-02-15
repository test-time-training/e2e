# End-to-End Test-Time Training for Long Context
[**Paper**](https://test-time-training.github.io/e2e.pdf)
| [**Setup**](#setup)
| [**Replicating Experiments**](#replicating-experiments)
| [**Model Checkpoints**](#model-checkpoints)

## Abstract

We formulate long-context language modeling as a problem in continual learning rather than architecture design.
Under this formulation, we only use a standard architecture – a Transformer with sliding-window attention.
However, our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights.
In addition, we improve the model's initialization for learning at test time via meta-learning at training time.
Overall, our method, a form of Test-Time Training (TTT), is End-to-End (E2E) both at test time (via next-token prediction) and training time (via meta-learning), in contrast to previous forms.


## Setup
This codebase is implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html) and has been tested on GPUs.

### Environment setup

We recommend the following **system GPU library** versions:

- **CUDA Toolkit** 12.8.1  
- **cuDNN** 9.8.0  
- **NCCL** 2.26.2 (built for CUDA 12.8)

We use **uv** for Python package management. Install `uv` with:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dataset Download
Our Llama-3 tokenized datasets are available for download from Google Cloud Storage buckets:
```
gcloud storage cp -r gs://llama3-dclm-filter-8k/ llama3-dclm-filter-8k
gcloud storage cp -r gs://llama3-books3/ llama3-books3
```
> Note (Requester Pays): These buckets may have Requester Pays enabled. If you encounter a billing/permissions error, follow [Google Cloud’s docs](https://docs.cloud.google.com/storage/docs/requester-pays).

Once downloaded, fill in the `deploy_paths` in `configs/deploy/interactive.yaml` (or `configs/deploy/submitit.yaml`). This will allow the dataloader to find the correct path. 

## Replicating Experiments

We use [Hydra](https://hydra.cc/) for configuration management. Configs for each experiment in the paper live under
`configs/experiment/`.

### Required Weights & Biases settings

Logging is done with Weights & Biases, and the following fields are required for launches below:

- `training.wandb_entity`
- `training.wandb_project`
- `training.wandb_key`

### Run on an interactive node

You can launch an experiment on an interactive node with:

```
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=my-entity \
  training.wandb_project=my-project \
  training.wandb_key=my-key
```

### Run multi-node on Slurm (Submitit)

For multi-node jobs on a Slurm cluster, we use Hydra’s Submitit launcher. For example:

```
uv run --exact train \
  +deploy=submitit \
  hydra.launcher.nodes=4 \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=my-entity \
  training.wandb_project=my-project \
  training.wandb_key=my-key
```

To configure additional Slurm settings (partition, account, GPUs per node, time limits, etc.), see
`configs/deploy/submitit.yaml` and the [Hydra Submitit Launcher docs](https://hydra.cc/docs/plugins/submitit_launcher/).

### Loading a model for extension

To initialize an extension run from a previous experiment, set:

- `training.resume_exp_name=<experiment_name>` to point to the experiment you want to resume from, and
- `training.load_part=params` to load model parameters from the most recent checkpoint.

On startup, the trainer will automatically locate the latest checkpoint in the experiment directory and restore it before beginning training.

## Model Checkpoints
We release a small set of checkpoints from the experiments in our paper. **Note:** this bucket has
[Requester Pays](https://cloud.google.com/storage/docs/requester-pays) enabled.

### Pre-trained (DCLM)

- **125M TTT-E2E (1× Chinchilla)** — DCLM @ **8K** context  
  [`gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc`](gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc)

- **1B TTT-E2E (1× Chinchilla)** — DCLM @ **8K** context  
  [`gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc`](gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc)

- **3B TTT-E2E (3× Chinchilla)** — DCLM @ **8K** context  
  [`gs://ttt-e2e-checkpoints/3b_ttt_e2e_pretrain_dclm_8k_3x_cc`](gs://ttt-e2e-checkpoints/3b_ttt_e2e_pretrain_dclm_8k_3x_cc)

### Extension fine-tuned (Books)

- **125M TTT-E2E (1× Chinchilla)** — Books @ **8K** context  
  [`gs://ttt-e2e-checkpoints/125m_ttt_e2e_finetune_books_8k_1x_cc`](gs://ttt-e2e-checkpoints/125m_ttt_e2e_finetune_books_8k_1x_cc)

- **1B TTT-E2E (1× Chinchilla)** — Books @ **8K** context  
  [`gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc`](gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc)

- **3B TTT-E2E (3× Chinchilla)** — Books @ **8K** context  
  [`gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_8k_3x_cc`](gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_8k_3x_cc)

- **3B TTT-E2E (3× Chinchilla)** — Books @ **128K** context  
  [`gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc`](gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc)

For usage, see [Loading a model for extension](#loading-a-model-for-extension). Optimizer state isn’t included, so `training.load_part=params` must be used.

If you’d like a checkpoint from another experiment in the paper, feel free to email us.