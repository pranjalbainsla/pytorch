# 09 — Coding Conventions (Core Systems)

These conventions are specific to `c10` / dispatcher / ATen / autograd / `nn` /
`optim`, plus repo norms that bite contributors in these trees. General process
(build, lint, CI): [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Design conventions by subsystem

### `torch.nn`

- **Modules are thin.** Prefer `return F.op(...)` in `forward`; do not reimplement
  numerics in Python that already exist as ATen ops.
- **Register state correctly.** Use `Parameter`, `Buffer`, or child `Module`
  assignments—not bare tensors—for anything that should appear in
  `parameters()` / `state_dict`.
- Call `super().__init__()` before assigning Parameters.
- Provide `reset_parameters()` and `extra_repr()` when adding a layer (match
  neighbors like `Linear`).
- Use `__constants__` for JIT-friendly hyperparameters when similar modules do.

### Autograd

- Prefer **CompositeImplicitAutograd** when a forward is a composition of
  existing differentiable ops ([04-aten-and-codegen.md](04-aten-and-codegen.md)).
- If you write a custom forward kernel, plan **`derivatives.yaml`** or a
  `torch.autograd.Function`—do not assume grads appear magically.
- Register autograd overrides on **`Autograd` / `AutogradXYZ`**, never as a
  substitute on the raw backend key ([03-dispatcher.md](03-dispatcher.md)).
- Save only what backward needs; saving too much burns memory.

### ATen / c10

- Edit **`native_functions.yaml` + native sources**, not generated files under
  the build tree.
- Copy **alias annotations** from a similar op; wrong `(a!)` is a silent
  footgun for views/inplace.
- Keep **dtype dispatch** (`AT_DISPATCH_*`) inside kernels; do not confuse it
  with DispatchKey registration.
- `native/cpu/` is for **ISA stubs**, not dumping every CPU kernel
  ([`native/cpu/README.md`](../aten/src/ATen/native/cpu/README.md)).
- c10 `impl/` has **no BC guarantees**—do not expose it as a public API.

### `torch.optim`

- New algorithms: **class + functional + single/foreach/(fused)**, following
  `adam.py` / `sgd.py` structure.
- Preserve **Tensor identity** for parameters; do not replace `param` with a
  new Tensor object in `step` (breaks `state` keying and Module ties).
- Prefer `zero_grad(set_to_none=True)` in examples and new code paths when
  equivalent.

## Repo style that matters here

From project agent guidance and existing code:

- **Minimize comments**; explain non-local context only.
- **ASCII only** in new comments (no smart quotes / em dashes).
- Prefer **clear state** on classes over dynamic `setattr`/`getattr` tricks
  (Modules already use structured dicts for a reason).
- Prefer keeping lines readable; if the linter wraps awkwardly, shorten names
  or use locals rather than sprawling multi-line expressions when possible.
- For golden string asserts that must stay one line: `# noqa: B950` on the
  closing quote line (see project docs)—not inside the string.
- Type stubs: edit **`.pyi.in`**, not generated `.pyi`, when applicable.

## Testing conventions

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    def test_foo(self):
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    run_tests()
```

- Use **`assertEqual`** for tensors (not raw `assert torch.allclose` alone in
  new PyTorch tests, unless matching nearby style).
- Use **`@parametrize`** for combinatorial inputs.
- Device-generic numeric tests: **`instantiate_device_type_tests`**.
- Autograd formulas: **`gradcheck` / `gradgradcheck`** where appropriate
  ([`torch/autograd/gradcheck.py`](../torch/autograd/gradcheck.py)).

## Linting and commits

- Lint via **`spin`** (`spin lint`, `spin fixlint`) per project norms.
- Before commit (when asked to commit): `lintrunner -a`.
- Do not touch [`.ci/docker/`](../.ci/docker/) unless you intend image rebuilds.

## Dynamo config (when tests need it)

Use `torch._dynamo.config.patch` as decorator/context manager—not manual
save/restore. (Compile itself is otherwise a black box in this guide.)

## What not to do

- Do not add a new Module that secretly calls NumPy for the hot path.
- Do not land huge formatting-only diffs in `module.py` / `TensorImpl.h`.
- Do not “fix” an autograd bug by disabling version checks.
- Do not register a CPU kernel at the `Autograd` key thinking it is “the”
  implementation—that skips or mishandles the layering.

## Related files

- [`aten/src/ATen/native/README.md`](../aten/src/ATen/native/README.md)
- [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [`torch/nn/modules/linear.py`](../torch/nn/modules/linear.py) (pattern reference)
- [`torch/optim/adam.py`](../torch/optim/adam.py) (pattern reference)

## Next

[10-performance.md](10-performance.md).
