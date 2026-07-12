# 12 — Historical Design Decisions

This page curates **design history documented in code comments and READMEs**,
not folklore. Knowing it prevents “why is it like this?” thrashing.

---

## Variable / Tensor merge

**Then:** `Variable` wrapped `Tensor` and carried autograd metadata.

**Now:** `using Variable = at::Tensor`; metadata is `AutogradMeta` on
`TensorImpl`. Python `torch.autograd.Variable` remains a legacy alias.

**Why it matters:** docs and APIs still say “Variable” in C++ autograd code;
you are looking at tensors. See [`variable.h`](../torch/csrc/autograd/variable.h).

---

## Function → Node rename

Graph vertices are C++ **`Node`**. Older docs/code said Function. Python still
exposes `torch.autograd.Function` (custom op) and `grad_fn` objects.

[`node.h`](../torch/csrc/autograd/node.h) comments describe the graph model
clearly—read them as current truth.

---

## THP prefix

From [`torch/csrc/autograd/README.md`](../torch/csrc/autograd/README.md):
**THP** = TorcH Python (`THPVariable`, `THPFunction`), not THPP.

Pattern: C++ payload type + Python object holding a pointer to it + sometimes a
`PyNode` bridging into Python callables.

---

## c10 naming and role

`c10` consolidated core types from the **Caffe2 + ATen** era into a minimal
library without kernels. ATen ops and autograd depend on it; it does not depend
on them.

---

## TH / THC heritage

[`aten/src/README.md`](../aten/src/README.md) records older TH tensor libraries
and refcounting rules. Modern code uses `intrusive_ptr` and `TensorImpl`. When
you see TH* names, treat them as legacy unless you are in an explicitly old
path.

---

## Autograd as an **alias** DispatchKey

**Note [Alias Dispatch Key : Autograd]** in
[`DispatchKey.h`](../c10/core/DispatchKey.h):

- Autograd is a **layer above backends**, not a property of CUDA itself.
- Default wrappers register to alias `Autograd`, expanding to AutogradCPU/CUDA/…
- Backend-specific autograd overrides use `AutogradXLA` etc.
- Registering “the math” at the raw `XLA` key does **not** replace Autograd
  (Autograd runs first).

This is why “I registered my kernel on CUDA but autograd still…” is a common
confusion—you likely wanted a different key.

---

## ADInplaceOrView and the VariableType “dream”

**Note [ADInplaceOrView key]** + **Note [Dream: skip VariableType…]**:

- Inplace/view bookkeeping is separated into `ADInplaceOrView`.
- Ideally, functional ops with `requires_grad=false` would skip Autograd
  wrappers entirely; in practice, dispatch-cost tradeoffs keep wrappers on the
  path, with guards excluding keys on redispatch.

History here explains residual overhead and surprising stack traces during
inference.

---

## CompositeImplicit / CompositeExplicit rename

From [`native/README.md`](../aten/src/ATen/native/README.md) history:

| Old | New |
|-----|-----|
| Math | CompositeImplicitAutograd |
| DefaultBackend | CompositeExplicitAutograd |

Implicit = decompose into differentiable ops (free grads). Explicit = need
explicit derivative formulas for autograd.

---

## Sharded VariableType

Autograd wrappers used to be one enormous generated file. They are now
**sharded** `VariableType_*.cpp` files (see templates under
[`tools/autograd/templates/`](../tools/autograd/templates/)). A combined file may
exist for convenience grepping—prefer the shard layout when debugging builds.

---

## RegisterOperators → TORCH_LIBRARY

Older registration API: `torch::RegisterOperators`.
New code: `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` ([`torch/library.h`](../torch/library.h)).
Both may appear while reading history.

---

## VariableFallbackKernel

[`VariableFallbackKernel.cpp`](../aten/src/ATen/core/VariableFallbackKernel.cpp)
exists so custom ops have *some* Autograd* behavior (fallthrough or not
implemented). Comments reference evolving custom-op autograd stories—prefer
explicit `autograd.Function` or proper formulas for real grads.

---

## Module + Functional split

`nn.Module` never owned numerics as a framework rule: layers wrap `F.*` so
stateful API and functional API stay aligned, and so autograd/ATen remain the
single numeric stack. Quantization builds **parallel module trees** rather than
branching every layer indefinitely.

---

## Optimizer foreach / fused evolution

Optimizers began as simple Python loops. **Foreach** batched operators reduced
Python overhead; **fused** kernels reduced launch overhead on GPU. The class /
functional split lets distributed and compiled trainers reuse the same math.

---

## Lab

1. Read [`torch/csrc/autograd/README.md`](../torch/csrc/autograd/README.md) fully.
2. Search `Note [` in [`DispatchKey.h`](../c10/core/DispatchKey.h) and skim each.
3. Find one `derivatives.yaml` entry and one CompositeImplicit op; contrast.

## Related files

- [`torch/csrc/autograd/variable.h`](../torch/csrc/autograd/variable.h)
- [`c10/core/DispatchKey.h`](../c10/core/DispatchKey.h)
- [`aten/src/ATen/native/README.md`](../aten/src/ATen/native/README.md)
- [`aten/src/README.md`](../aten/src/README.md)

## Next

[13-common-pitfalls.md](13-common-pitfalls.md).
