import sys, os, json
sys.path.insert(0, "/w/tools/kernels")
import torch, k3_harness as H
gdl = H.load_gdl()
dev = torch.device("cuda")
gen = torch.Generator().manual_seed(0)
L = H.make_layer(torch, dev, 8, 3, gen)
W = 4; c = 8
nslots = 1 + c*W + 4
st0 = (torch.randn(nslots, L["HV"], 128, 128, generator=gen) * 0.05).to(dev)
s_s, s_l = st0.clone(), st0.clone()
hdr = torch.zeros(nslots, dtype=torch.int32, device=dev)
si = torch.arange(1, 1 + c*W, dtype=torch.int32, device=dev).view(c, W)
acc = torch.ones(c, dtype=torch.int32, device=dev)
lens = [W]*c
tot = 0; shown = 0
for step in range(int(os.environ.get("STEPS", "4"))):
    mixed, a, b, gate = H.make_tokens(torch, dev, L, lens, gen)
    cu = torch.tensor([0] + list(H._cumsum(lens)), dtype=torch.int32, device=dev)
    seq = torch.full((c,), 1000, dtype=torch.int32, device=dev)
    o_s = torch.zeros(mixed.shape[0], L["HV"], 128, dtype=torch.bfloat16, device=dev); o_l = torch.zeros_like(o_s)
    H.stock_cuda(mixed_qkv=mixed, a=a, b=b, A_log=L["A_log"], dt_bias=L["dt_bias"], state_indices=si, cu_seqlens=cu,
                 num_accepted_tokens=acc, state=s_s, output_gate=gate, norm_weight=L["norm_w"], out=o_s,
                 scale=128 ** -0.5, norm_eps=1e-6, output_gate_activation="sigmoid")
    gdl.gdl_decode_launch(mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq, s_l, hdr, gate, L["norm_w"], o_l, 8, 128 ** -0.5, 1e-6, True,
                          mode=int(os.environ.get("MODE", "0")))
    d = (o_s.view(torch.int16) != o_l.view(torch.int16))
    tot += int(d.sum())
    for idx in d.nonzero().tolist()[:6]:
        t, hv, j = idx
        print(step, "tok", t, "hv", hv, "dim", j, float(o_s[t, hv, j]), float(o_l[t, hv, j]),
              "rowdiffs", int(d[t, hv].sum()), "rms", float(o_s[t,hv].float().pow(2).mean().sqrt()))
    acc = torch.full((c,), 2, dtype=torch.int32, device=dev)
print("total diffs", tot)
k = gdl._gdl_decode_kernel
try:
    cache = k.device_caches[0][0] if hasattr(k, "device_caches") else k.cache[0]
    for key, ck in list(cache.items())[:1]:
        open("/s/decode.ttgir", "w").write(ck.asm["ttgir"]); open("/s/decode.sass", "w").write(ck.asm.get("sass", "") or "")
        open("/s/decode.ptx", "w").write(ck.asm["ptx"])
        print("dumped", ck.n_regs, ck.n_spills)
except Exception as e:
    print("dump failed", e)
