# c3 MTP FP8_BLOCK_SCALES overlay (reverted)

Skeptic rejected this candidate. It was not a new pin. `docker/modelopt.py` already shipped in c1. c2 kept the same digest without `VLLM_PLE_FP8_CHECKPOINT`.

Stop predicate for this revert. Tree pin files match `60bc4af`. Live serve stays the c2 container. `GET /health` 200. Do not reboot. Do not delete the c1 overlay.
