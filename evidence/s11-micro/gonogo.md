# S1.1 go/no-go

## pin

| Measurement | Value | Decision rule | Verdict | Note |
|---|---|---|---|---|
| In-graph AR 5-160 KB, mixing on vs off (L2a booking) | dAR(M) us 1:20.1, 4:20.6, 8:18.6, 16:17.7, 32:24.4; booked c1 1.10 ms, c8 1.29 ms | book 0.5 x (107 x dAR + neighbour slowdown); build L2a if >= 0.75 ms | GO | gapped mode already contains the slowed neighbour |
| AG vs AR at c=1 payload, mixing off (L2d) | AG 16.5 / AR 21.0 us = 0.79 at M=4 | build L2d only if AG_off <= 0.7 x AR_off | NO-GO |  |
| L2 prefetch of a 13.4 MB module during a real AR | -3.1 us/window at M=4 | >= 15 us net per window -> revisit F16; otherwise 0 | NO-GO |  |
| Logits all-gather 248 KB-8 MB (keep, gapped) | 242 KB:80 us, 970 KB:213 us, 1940 KB:285 us, 3880 KB:439 us, 7760 KB:836 us | input to F13 / L1 comm model | INFO |  |
| cuBLAS choice for HC down (336,10240) and HC up at M=4 | down 35.8 us [void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616g + void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat1]; up 30.7 us [void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616g] (split-K kernel seen) | WMMA tile (>= 60 us per call) -> L4a first; split-K (~37 us) -> K5 directly, built W8A16-ready | GO | split-K case: skip L4a, build K5 directly (W8A16-ready) |
| cuBLAS GB/s at (8240,2560) (gdn_in_proj_qkvz_ba) | 221.5 GB/s at M=4 cold (M=1:168.3, M=32:215.4) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (2560,3072) (gdn_out_proj/qsa_o) | 219.9 GB/s at M=4 cold (M=1:163.4, M=32:212.1) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (7296,2560) (qsa_qkv+index) | 228.6 GB/s at M=4 cold (M=1:169.9, M=32:223.6) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (48,2560) (gdn_in_proj_ba) | 12.6 GB/s at M=4 cold (M=1:59.1, M=32:34.0) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (512,2560) (router) | 148.4 GB/s at M=4 cold (M=1:182.9, M=32:161.9) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | GO | GO = a K4 BF16 plan pays for this shape |
| w8a16_marlin_ch GB/s at the P4-1 shapes | worst 192.0 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:217.9, gdn_out_proj/qsa_o:192.0, qsa_qkv:196.8, qsa_qkv+index:207.3, ple_kv_proj:222.3 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | GO | GO = stock kernel; NO-GO = K4 FP8 first |
| w8a16_marlin_b128 GB/s at the P4-1 shapes | worst 206.0 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:225.9, gdn_out_proj/qsa_o:206.0, qsa_qkv:212.8, qsa_qkv+index:212.5, ple_kv_proj:220.7 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | GO | GO = stock kernel; NO-GO = K4 FP8 first |
| w8a8_cutlass GB/s at the P4-1 shapes | worst 174.5 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:200.7, gdn_out_proj/qsa_o:174.5, qsa_qkv:196.0, qsa_qkv+index:195.5, ple_kv_proj:207.9 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | MARGINAL | GO = stock kernel; NO-GO = K4 FP8 first |
| D8 arm order (W8A8 >= 200 GB/s?) | W8A8 worst 174.5 GB/s | W8A16 first unless prefill TTFT is the priority and W8A8 >= 200 | INFO | W8A16 arm first |
| NVFP4 MoE TP2 routed vs floor | 296.0 us vs floor 240.4 us = 1.231x (40 experts, M=4, autotuned=True); EP 1.133x | TP2 routed <= 1.2x floor -> close F10 | NO-GO | GO = close F10 |
| NVFP4 MoE TP2 FC2 vs floor | FC2 100.35 us vs 80.14 us = 1.252x | TP2 FC2 >= 1.6x floor -> run the D2 `ep` A/B | NO-GO | GO = run D2 ep A/B |
| L6b BF16 activations into FlashInfer | 294.9 us vs fp4_in 296.0 us | informs L6b (-0.1 ms expected) | INFO |  |
| Draft-head probe F.linear [V'/2,2560] + argmax | M=1 cold: V'=32768:166.6 GB/s 503 us, V'=47104:174.8 GB/s 690 us, V'=65536:172.0 GB/s 975 us, V'=248320:171.6 GB/s 3705 us | >= 200 GB/s -> L1b is on track | NO-GO | rule applied at V'=32k |
| 42 MB all-reduce, single vs dual rail | single 3724 us, dual 2008 us (1.855x); worst small-msg latency +107.2% | feeds longctx only; records the decode small-message penalty | INFO |  |
| deviceQuery facts (spark1) | sm_count=48, l2_bytes=25165824, smem_per_block_optin=101376 | 48 SMs, 24 MB L2, ~99 KB opt-in smem | INFO | kernel track (§4) assumptions |
| HC module chain baseline (L4 gate: <= 0.6x this) | M=4 stream 71.8 us (1.23x floor), hot 24.7 us, flush 68.6 us | baseline for L4 / K5 | INFO |  |

## v030

| Measurement | Value | Decision rule | Verdict | Note |
|---|---|---|---|---|
| In-graph AR 5-160 KB, mixing on vs off (L2a booking) | dAR(M) us 1:16.4, 4:16.6, 8:14.5, 16:14.4, 32:19.3; booked c1 0.89 ms, c8 1.02 ms | book 0.5 x (107 x dAR + neighbour slowdown); build L2a if >= 0.75 ms | GO | gapped mode already contains the slowed neighbour |
| AG vs AR at c=1 payload, mixing off (L2d) | AG 15.8 / AR 20.9 us = 0.76 at M=4 | build L2d only if AG_off <= 0.7 x AR_off | NO-GO |  |
| L2 prefetch of a 13.4 MB module during a real AR | -2.4 us/window at M=4 | >= 15 us net per window -> revisit F16; otherwise 0 | NO-GO |  |
| Logits all-gather 248 KB-8 MB (keep, gapped) | 242 KB:75 us, 970 KB:211 us, 1940 KB:283 us, 3880 KB:434 us, 7760 KB:830 us | input to F13 / L1 comm model | INFO |  |
| cuBLAS choice for HC down (336,10240) and HC up at M=4 | down 34.8 us [void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616g + void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat1]; up 29.3 us [void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616g] (split-K kernel seen) | WMMA tile (>= 60 us per call) -> L4a first; split-K (~37 us) -> K5 directly, built W8A16-ready | GO | split-K case: skip L4a, build K5 directly (W8A16-ready) |
| cuBLAS GB/s at (8240,2560) (gdn_in_proj_qkvz_ba) | 222.7 GB/s at M=4 cold (M=1:168.2, M=32:215.7) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (2560,3072) (gdn_out_proj/qsa_o) | 219.4 GB/s at M=4 cold (M=1:160.8, M=32:211.1) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (7296,2560) (qsa_qkv+index) | 225.7 GB/s at M=4 cold (M=1:168.1, M=32:220.4) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | NO-GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (48,2560) (gdn_in_proj_ba) | 12.4 GB/s at M=4 cold (M=1:58.9, M=32:34.3) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | GO | GO = a K4 BF16 plan pays for this shape |
| cuBLAS GB/s at (512,2560) (router) | 154.0 GB/s at M=4 cold (M=1:181.4, M=32:159.7) | < 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate | GO | GO = a K4 BF16 plan pays for this shape |
| w8a16_marlin_ch GB/s at the P4-1 shapes | worst 198.3 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:213.5, gdn_out_proj/qsa_o:198.3, qsa_qkv:210.1, qsa_qkv+index:212.8, ple_kv_proj:222.2 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | GO | GO = stock kernel; NO-GO = K4 FP8 first |
| w8a16_marlin_b128 GB/s at the P4-1 shapes | worst 215.4 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:221.0, gdn_out_proj/qsa_o:215.4, qsa_qkv:217.9, qsa_qkv+index:217.1, ple_kv_proj:226.8 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | GO | GO = stock kernel; NO-GO = K4 FP8 first |
| w8a8_cutlass GB/s at the P4-1 shapes | worst 170.8 GB/s (gdn_out_proj/qsa_o) at M=4; gdn_in_proj_qkvz:209.7, gdn_out_proj/qsa_o:170.8, qsa_qkv:198.0, qsa_qkv+index:200.4, ple_kv_proj:210.7 | >= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1 | MARGINAL | GO = stock kernel; NO-GO = K4 FP8 first |
| D8 arm order (W8A8 >= 200 GB/s?) | W8A8 worst 170.8 GB/s | W8A16 first unless prefill TTFT is the priority and W8A8 >= 200 | INFO | W8A16 arm first |
| NVFP4 MoE TP2 routed vs floor | 297.6 us vs floor 240.4 us = 1.238x (40 experts, M=4, autotuned=True); EP 1.134x | TP2 routed <= 1.2x floor -> close F10 | NO-GO | GO = close F10 |
| NVFP4 MoE TP2 FC2 vs floor | FC2 99.94 us vs 80.14 us = 1.247x | TP2 FC2 >= 1.6x floor -> run the D2 `ep` A/B | NO-GO | GO = run D2 ep A/B |
| L6b BF16 activations into FlashInfer | 293.6 us vs fp4_in 297.6 us | informs L6b (-0.1 ms expected) | INFO |  |
| Draft-head probe F.linear [V'/2,2560] + argmax | M=1 cold: V'=32768:169.5 GB/s 495 us, V'=47104:168.9 GB/s 714 us, V'=65536:170.8 GB/s 982 us, V'=248320:168.6 GB/s 3770 us | >= 200 GB/s -> L1b is on track | NO-GO | rule applied at V'=32k |
| 42 MB all-reduce, single vs dual rail | single 3641 us, dual 2004 us (1.817x); worst small-msg latency +113.9% | feeds longctx only; records the decode small-message penalty | INFO |  |
| deviceQuery facts (spark1) | sm_count=48, l2_bytes=25165824, smem_per_block_optin=101376 | 48 SMs, 24 MB L2, ~99 KB opt-in smem | INFO | kernel track (§4) assumptions |
| HC module chain baseline (L4 gate: <= 0.6x this) | M=4 stream 70.8 us (1.212x floor), hot 24.7 us, flush 74.5 us | baseline for L4 / K5 | INFO |  |
