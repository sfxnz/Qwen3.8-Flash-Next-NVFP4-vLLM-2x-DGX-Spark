IMG=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
for seed in 1 2 3; do
 docker run --rm --gpus all --ipc host --network none --ulimit core=1 --memory 24g -v $PWD:/w -w /w --entrypoint python3 $IMG tools/kernels/k3_harness.py exact --conc 1 2 5 8 --steps 1024 --seed $seed | tr -d ' \n' | grep -o '"c[0-9]*".\{0,0\}\|"A0":[a-z]*' | tr '\n' ' '; echo " exact seed $seed"
 docker run --rm --gpus all --ipc host --network none --ulimit core=1 --memory 24g -v $PWD:/w -w /w --entrypoint python3 $IMG tools/kernels/k3_harness.py sim --block 64 --nreq 8 --steps 600 --seed $seed 2>&1 | tail -1; echo " sim64 rc=${PIPESTATUS[0]} seed $seed"
done
