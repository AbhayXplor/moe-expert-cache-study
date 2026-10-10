#!/usr/bin/env python
"""profile_compress.py -- where does DF11 compression time actually go?

Runs the released compress_model on the first 2 modules of a small model under
cProfile and prints the top cumulative entries.
"""
import cProfile, importlib.util, io, os, pstats, shutil

spec = importlib.util.spec_from_file_location("cdf", "scripts/compress_df11.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.quiet_dfloat11()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from dfloat11.dfloat11 import compress_model  # noqa: E402

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).eval()
pattern_dict, prefix, common = mod.discover_pattern_dict(model)
print("[profile] pattern:", pattern_dict, flush=True)

out = "experiments/_profile_tmp"
shutil.rmtree(out, ignore_errors=True)
pr = cProfile.Profile()
pr.enable()
compress_model(model, pattern_dict, out, block_range=[0, 2], save_single_file=False,
               check_correctness=True)
pr.disable()

buf = io.StringIO()
st = pstats.Stats(pr, stream=buf)
st.sort_stats("cumulative").print_stats(20)
print(buf.getvalue())
shutil.rmtree(out, ignore_errors=True)
