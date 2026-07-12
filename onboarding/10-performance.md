# 10 — Performance Considerations

Scope: performance levers inside c10 / dispatcher / ATen / autograd / nn /
optim. Compiler stacks (`torch.compile`) are mentioned only as boundaries.

## Hot-path mental model

Almost every training op pays for:

1. Python binding → `at::op`
2. **Dispatcher** lookup
3. **Autograd wrapper** (even when many inputs do not need grad—see dream note)
4. Numeric kernel (TensorIterator / cuBLAS / fused / …)
5. Optional graph Node allocation and **SavedVariable** storage

Inference without grad mode still often traverses Autograd wrappers that
redispatch quickly; true “skip VariableType when possible” remains a documented
aspiration in [`DispatchKey.h`](../c10/core/DispatchKey.h)
(**Note [Dream: skip VariableType kernel when requires_grad=false]**).

## Autograd costs

| Cost | Mitigation |
|------|------------|
| Building Nodes every forward | `torch.no_grad()` / `inference_mode()` for pure inference |
| Saved tensors memory | Gradient checkpointing (activation recomputation)—conceptual; impl may live outside this folder |
| Inplace that forces versions / copies | Prefer out-of-place unless carefully aliased |
| Python `Function` | Extra Python roundtrips vs generated C++ Nodes—use for correctness first |

Anomaly mode and too many hooks also slow backward—use for debug only.

## Layout and memory

- **Contiguous** inputs often hit faster kernels; unexpected non-contiguity
  causes implicit `contiguous()` copies inside ops.
- **Channels-last** can help some conv CUDA paths—measure; do not assume.
- **Views** are cheap; **materializing** copies is not. Know when an op returns
  a view vs a new storage ([02-c10-tensor-and-storage.md](02-c10-tensor-and-storage.md)).
- Avoid accidental **dtype/device** mismatches that force casts/transfers.

## TensorIterator and elementwise ops

TensorIterator handles broadcasting and promotion—a productivity win with
overhead. For tiny ops in a tight Python loop, overhead dominates; for large
tensors, kernel time dominates. When writing a new elementwise op, follow
existing TensorIterator patterns rather than inventing a one-off.

## CPU ISA stubs

Hot CPU kernels use `DispatchStub` + AVX2/AVX512 variants
([`native/cpu/README.md`](../aten/src/ATen/native/cpu/README.md)). If your op is
on the critical path on CPU, check whether neighbors use stubs.

## Optimizer performance

```mermaid
flowchart LR
  slow["single_tensor Python loop"]
  mid["foreach _foreach_*"]
  fast["fused CUDA kernel"]
  slow -->|"many params"| mid -->|"GPU training"| fast
```

- Default Adam/SGD paths prefer **foreach** when safe.
- **`fused=True`** fuses elementwise update math on supporting devices—big wins
  with huge parameter counts.
- **`zero_grad(set_to_none=True)`** avoids writing zeros into grad storages.
- Do not step under unnecessary grad recording (`differentiable=True` is a
  special case and slower).

See [07-optimizers.md](07-optimizers.md).

## `nn.Module` overhead

- Hook-less modules use a **fast path** in `_call_impl`. Hooks (including
  global ones) disable it—expensive if applied to every layer in a huge model.
- Deep Module trees have Python call overhead; this is usually dwarfed by
  kernels except for tiny models / microbenchmarks.
- `extra_repr` / printing huge modules can be slow—irrelevant to training step
  time but annoying in logs.

## Dispatcher / registration pitfalls that look like “perf bugs”

- Missing kernel → unexpected fallthrough / CPU fallback → silent slowdown.
- Forcing sync (`tensor.item()`, `.cpu()` on CUDA tensors) in a loop.
- Logging that copies tensors every iteration.

## Measuring

Within scope of core contributions:

- Microbenchmarks next to similar ops (see existing `*-bench` patterns in-tree).
- Compare foreach vs fused vs single for optim changes.
- For autograd formula changes: correctness via `gradcheck` first; then memory
  (`saved_tensors`) if relevant.

Keep `torch.compile` comparisons as a separate experiment—the generated graph
may remove Python overhead you were profiling.

## Lab

1. Time Adam `foreach=False` vs default vs `fused=True` on CUDA with a large
   `nn.Sequential` of Linears.
2. Forward under `inference_mode()` vs default grad mode; compare time for a
   fixed model.
3. Transpose a large tensor without `.contiguous()` and pass into an op; check
   whether a copy appears (profiler or `torch.autograd.profiler`).

## Related files

- [`c10/core/DispatchKey.h`](../c10/core/DispatchKey.h) (ADInplaceOrView / dream notes)
- [`torch/nn/modules/module.py`](../torch/nn/modules/module.py) (`_call_impl` fast path)
- [`torch/optim/optimizer.py`](../torch/optim/optimizer.py)
- [`aten/src/ATen/TensorIterator.h`](../aten/src/ATen/TensorIterator.h)

## Next

[11-extension-points.md](11-extension-points.md).
