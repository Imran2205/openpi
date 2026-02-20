"""Compute normalization statistics for a config.

This is a safer variant of `compute_norm_stats.py`.

In addition to computing norm stats, this script can sanitize quantile stats for
near-constant dimensions (commonly action padding or rarely-changing channels) to
avoid unstable quantile normalization.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def sanitize_quantile_stats(
    norm_stats: dict[str, normalize.NormStats],
    *,
    keys: tuple[str, ...] = ("actions",),
    min_q_span: float = 1e-3,
    min_std: float = 1e-6,
    fallback_half_span: float = 1.0,
) -> tuple[dict[str, normalize.NormStats], dict[str, list[int]]]:
    """Sanitize quantile stats for near-constant dimensions.

    For dimensions where q99-q01 is too small (or std is ~0), quantile normalization
    can amplify tiny noise and destabilize training. For those dimensions we replace
    q01/q99 with a safe fallback interval centered at the mean.
    """
    sanitized = dict(norm_stats)
    changed_dims: dict[str, list[int]] = {}

    for key in keys:
        stats = sanitized.get(key)
        if stats is None or stats.q01 is None or stats.q99 is None:
            continue

        mean = np.asarray(stats.mean, dtype=np.float64).copy()
        std = np.asarray(stats.std, dtype=np.float64).copy()
        q01 = np.asarray(stats.q01, dtype=np.float64).copy()
        q99 = np.asarray(stats.q99, dtype=np.float64).copy()

        span = q99 - q01
        bad = (~np.isfinite(span)) | (~np.isfinite(std)) | (span < min_q_span) | (std < min_std)
        bad_idx = np.where(bad)[0].tolist()

        if bad_idx:
            half_span = np.maximum(fallback_half_span, 3.0 * np.maximum(std[bad], min_std))
            q01[bad] = mean[bad] - half_span
            q99[bad] = mean[bad] + half_span
            std[bad] = np.maximum(std[bad], min_std)
            changed_dims[key] = bad_idx

        sanitized[key] = normalize.NormStats(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            q01=q01.astype(np.float32),
            q99=q99.astype(np.float32),
        )

    return sanitized, changed_dims


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    sanitize_keys: tuple[str, ...] = ("actions",),
    min_q_span: float = 1e-3,
    min_std: float = 1e-6,
    fallback_half_span: float = 1.0,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    norm_stats, changed_dims = sanitize_quantile_stats(
        norm_stats,
        keys=sanitize_keys,
        min_q_span=min_q_span,
        min_std=min_std,
        fallback_half_span=fallback_half_span,
    )
    if changed_dims:
        print("Sanitized near-constant dims for quantile stats:")
        for key, dims in changed_dims.items():
            print(f"  {key}: {dims}")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
