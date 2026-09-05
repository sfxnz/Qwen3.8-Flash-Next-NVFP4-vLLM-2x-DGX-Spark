# c5 MTP k=2 on prose

## Mechanism

Live k=3 prose acceptance_len on the last kept pin (c4 native 262144) is 2.46 at c=1 and 1.70 at c=2. Extra rejected drafts sit past the mean. Each extra speculative token still rebuilds QSA draft metadata.

k=2 caps acceptance_len at 3.0 (1 plus 2 drafts). Structured c=1 on c4 is lossless at 4.0, so k=2 can cut structured tok/s. This is one knob after the NVIDIA pin holds.

## Ruler

Frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200`. Same file hash as `evidence/opt-c4-native-ctx-262144/harness.sha256`. Compare to last kept pin c4, not the stale RadixArk opt-baseline. Occupancy probe is c=8 because the pin claims seqs=8.

## Stop

One attempt. Keep if health 200, quality smokes pass, seqs=8 does not drop a stream, at least one of concurrency / decode prose / TTFT improves past noise versus c4, and none of the five bars regress past noise. Keep k=3 if structured tok/s falls past noise.
