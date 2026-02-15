import grain.python as grain
import jax
import numpy as np
import zarr.codecs
import zarr.storage

from ttt.model.data import Batch


class Dataset(grain.RandomAccessDataSource):
    def __init__(self, *, path: str, split: str, seq_len: int):
        codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)

        store = zarr.storage.LocalStore(path, read_only=True)

        self._dataset = zarr.open_array(store, path=f"/{split}", codec=codec)

        self.split = self._dataset
        self.seq_len = seq_len

    def __getitem__(self, idx):
        sample = self.split[idx * self.seq_len : (idx + 1) * self.seq_len + 1]
        assert len(sample) == (self.seq_len + 1), "Loader got a sequence with the wrong length!"
        return sample

    def __len__(self):
        return (self.split.shape[0] - 1) // self.seq_len


class DummyDataset(grain.RandomAccessDataSource):
    def __init__(self, *, seq_len: int, num_tokens: int = 2**25):
        self.seq_len = seq_len
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        sample = np.random.randint(0, 20, (self.seq_len + 1,), dtype=np.int32)
        return sample

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len


def _to_batch(
    data: np.ndarray,
    *,
    bos_token_id: int,
    eos_token_id: int,
) -> Batch:
    tokens = np.asarray(data)
    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )


def lm_dataset(
    *,
    path: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
) -> grain.MapDataset:
    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()

    assert global_batch_size % shard_count == 0
    host_batch_size = global_batch_size // shard_count

    source = Dataset(path=path, split=split, seq_len=seq_len)
    dataset = grain.MapDataset.source(source)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    dataset_length = len(source)

    if repeat:
        print(f"Repeating dataset. Length {dataset_length}.")
        dataset = dataset.repeat()
    else:
        dataset_length = len(dataset)
        trimmed_length = (dataset_length // shard_count) * shard_count  # Drop remainder
        dataset = dataset[:trimmed_length]
        print(f"Trimming dataset. Initial length {dataset_length}. New length {trimmed_length}.")

    dataset = dataset[shard_index::shard_count]

    return dataset


def dummy_dataset(
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    repeat: bool = False,
    num_tokens: int = 2**25,
):
    shard_index = jax.process_index()
    shard_count = jax.process_count()

    dataset = grain.MapDataset.source(
        DummyDataset(seq_len=seq_len, num_tokens=num_tokens),
    )

    host_batch_size = global_batch_size // shard_count
    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    if repeat:
        print("Repeating dataset.")
        dataset = dataset.repeat()

    dataset = dataset[shard_index::shard_count]
    return dataset
