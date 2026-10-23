# Checkpoint compatibility

Verified on 2026-10-23 with the original HRNet-W32 experiment checkpoint.

| Check | Result |
| --- | --- |
| Complete original model checkpoint loading | `strict=True`, passed |
| Complete Glimpse model checkpoint loading | `strict=True`, passed |
| State dictionary keys | 1,999; identical |
| Missing / unexpected keys | 0 / 0 |
| Same synthetic input through both complete models | Exact equality; maximum absolute difference 0.0 |
| Output shape | `[1, 1, 17, 3]` |
| Environment | Python 3.12.9, PyTorch 2.4.1+cu121, CPU |

The model dictionary in the checkpoint is stored under `model`. The existing loader strips `module.` from distributed-training keys and loads it strictly. Class names are not part of these state dictionary keys, so a checkpoint conversion is unnecessary.

SHA-256 of the unchanged release asset `glimpse_hrnet32.bin`:

```text
644e03584fb6dbfd4fc6f10b82b43be826ff325363ed9dd797075e7582b58649
```

The public model files preserve the renamed server sources, apart from Git line-ending normalization. The public training entry differs from the renamed server entry only in resource path strings. The YAML uses portable resource paths. No model arithmetic, preprocessing, optimization, flip testing or metric computation was changed for publication.

These checks establish compatibility and output equivalence. They are not a rerun of the full Human3.6M benchmark, nor a verification that every reported paper result is reproduced by this particular checkpoint.
