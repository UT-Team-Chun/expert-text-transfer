"""National-scale geotechnical foundation model package.

See ``docs/architecture.md`` for the high-level design. Modules:

- ``national.data`` -- covariate registry and boring datasets.
- ``national.tiling`` -- regional tiles, halos, regime classifier.
- ``national.models`` -- DKL+SVGP foundation model and online conditioner.
- ``national.training`` -- Hydra-driven distributed training driver.
- ``national.prediction`` -- tiled inference engine, Zarr/COG output.
- ``national.evaluation`` -- spatial K-fold, LRO, calibration, baselines.
- ``national.api`` -- FastAPI prediction endpoints.
"""

__all__: list[str] = []
