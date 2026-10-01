# variant.py BV NW CMD...: run k3_harness with a patched copy of gdn_lazy.py (GDL_BV, decode num_warps)
import sys, re
from pathlib import Path
sys.path.insert(0, "/w/tools/kernels")
import k3_harness as H
bv, nw = sys.argv[1], sys.argv[2]
src = Path("/w/docker/v030/gdn_lazy.py").read_text()
src = src.replace("GDL_BV = 32 ", f"GDL_BV = {bv} ")
src = src.replace("RING_OFF=GDL_RING_OFF, RED=_gdl_red(),\n        num_warps=4,", f"RING_OFF=GDL_RING_OFF, RED=_gdl_red(),\n        num_warps={nw},", 1)
p = Path(f"/tmp/gdl_{bv}_{nw}.py"); p.write_text(src)
H.KERNELS = p
sys.exit(H.main(sys.argv[3:]))
