# 06 — `torch.nn`: Modules and Functional

## Theory first

Parameters vs buffers vs activations:
[01-theory-primer.md](01-theory-primer.md)#6-parameters-vs-buffers-vs-activations-nn-theory.

**Design rule:** Modules hold state and hyperparameters; **numerics live in
stateless ops** (`nn.functional` → ATen). Autograd does not special-case
Modules—it only sees tensors and ops.

## Package layout

| Path | Role |
|------|------|
| [`torch/nn/__init__.py`](../torch/nn/__init__.py) | Public surface |
| [`torch/nn/modules/module.py`](../torch/nn/modules/module.py) | **`Module` base** |
| [`torch/nn/modules/`](../torch/nn/modules/) | Stateful layers |
| [`torch/nn/functional.py`](../torch/nn/functional.py) | Stateless `F.*` |
| [`torch/nn/parameter.py`](../torch/nn/parameter.py) | `Parameter`, `Buffer` |
| [`torch/nn/init.py`](../torch/nn/init.py) | Initializers |
| [`torch/nn/modules/container.py`](../torch/nn/modules/container.py) | Sequential, ModuleList, … |
| [`torch/nn/utils/`](../torch/nn/utils/) | clip_grad, parametrize, prune, … |

There is **no README inside `torch/nn/`**. Sphinx docs live under `docs/source/nn*.md`.

## What `Module` stores

On `__init__` (simplified):

```text
_parameters, _buffers, _modules
_forward_pre_hooks / _forward_hooks
_backward_pre_hooks / _backward_hooks
_state_dict_* / _load_state_dict_* hooks
training: bool
```

### Registration via `__setattr__`

| Assignment | Effect |
|------------|--------|
| `self.w = Parameter(...)` | `register_parameter` → `_parameters` |
| `self.sub = Module(...)` | `_modules[name] = sub` |
| `self.b = Buffer(...)` / buffer overwrite | `register_buffer` → `_buffers` |
| plain `Tensor` / other | normal attribute — **not** tracked as param/buffer |

`__getattr__` looks up `_parameters` → `_buffers` → `_modules` so registered
names behave like attributes without living on `__dict__` as ordinary fields.

**Pitfall:** `self.weight = torch.empty(...)` does **not** register a parameter.
Use `Parameter` or `register_parameter`.

## Call path: `__call__` → `forward`

```mermaid
flowchart TD
  call["module(*args, **kwargs)"]
  wrap["_wrapped_call_impl"]
  compiled["_compiled_call_impl if torch.compile hooked"]
  impl["_call_impl"]
  fast["fast path: forward(*args)"]
  hooks["forward_pre_hooks"]
  fwd["forward / _slow_forward"]
  fhooks["forward_hooks"]
  bsetup["BackwardHook setup"]
  call --> wrap
  wrap --> compiled
  wrap --> impl
  impl --> fast
  impl --> hooks --> fwd --> fhooks
  hooks --> bsetup
  fhooks --> bsetup
```

Important details from [`module.py`](../torch/nn/modules/module.py):

- `__call__` is aliased to `_wrapped_call_impl`.
- If no hooks (local or global), **fast path** calls `forward` directly.
- Else: global then local **forward_pre_hooks** (may rewrite args), then
  `forward`, then **forward_hooks** (may replace result), plus backward hook
  wiring.
- Default `forward` raises if not overridden (`_forward_unimplemented`).
- JIT tracing may use `_slow_forward`.
- `torch.compile` may install `_compiled_call_impl` (black box for this guide).

## Canonical pattern: `Linear`

From [`torch/nn/modules/linear.py`](../torch/nn/modules/linear.py):

```python
self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
# ...
def forward(self, input: Tensor) -> Tensor:
    return F.linear(input, self.weight, self.bias)
```

`F.linear` is a thin documented binding into ATen. Gradients come from ATen /
autograd—not from Module code.

| Layer file | Module | Functional |
|------------|--------|------------|
| `linear.py` | `Linear` | `F.linear` |
| `conv.py` | `Conv2d` | `F.conv2d` |
| `batchnorm.py` | `BatchNorm*` | `F.batch_norm` |
| `activation.py` | `ReLU` | `F.relu` |
| `loss.py` | `CrossEntropyLoss` | `F.cross_entropy` |

## Parameters and buffers

[`torch/nn/parameter.py`](../torch/nn/parameter.py):

- **`Parameter`**: `Tensor` subclass (`_make_subclass`); assignment to Module
  auto-registers; typically `requires_grad=True`.
- **`Buffer`**: similarly auto-registers as buffer; `persistent=` controls
  `state_dict` inclusion.
- **`UninitializedParameter` / `UninitializedBuffer`**: lazy shapes with
  [`LazyModuleMixin`](../torch/nn/modules/lazy.py).

## Tree operations

Recursive over `_modules`:

- `parameters()` / `named_parameters()`, `buffers()` / `named_buffers()`
- `children()`, `modules()`, `named_modules()`
- `train()` / `eval()` flip `training` recursively
- `to()` / `cuda()` / `cpu()` / `to_empty()` via `_apply`
- `state_dict` / `load_state_dict` (+ optional `get_extra_state`)

## Containers

[`container.py`](../torch/nn/modules/container.py): `Sequential`, `ModuleList`,
`ModuleDict`, `ParameterList`, `ParameterDict`. They exist so nested modules
register correctly (a plain Python list of Modules would **not**).

## Hooks as extension middleware

| Hook | When |
|------|------|
| `register_forward_pre_hook` | Before `forward` |
| `register_forward_hook` | After `forward` |
| `register_full_backward_hook` | During backward on the module |
| Global `register_module_*_hook` | All modules |

Also: state_dict load/save hooks; module registration hooks.

Used by weight norm, spectral norm, parametrizations
([`torch/nn/utils/parametrize.py`](../torch/nn/utils/parametrize.py)), profiling, etc.

## Init

[`torch/nn/init.py`](../torch/nn/init.py): `kaiming_*`, `xavier_*`, `orthogonal_`,
… typically under `no_grad`. Layers call these from `reset_parameters()`.

## Quantization / parallel (black box note)

`quantized/`, `qat/`, `parallel/` exist as parallel trees. Treat as out of
scope unless your work lands there; the Module registration model is the same.

## Lab

1. Read `_call_impl` in [`module.py`](../torch/nn/modules/module.py) (~1782+).

2. Prove registration:

   ```python
   import torch.nn as nn
   m = nn.Linear(3, 4)
   print(list(m._parameters.keys()))
   m.foo = torch.zeros(1)  # NOT a parameter
   print("foo" in m._parameters)
   ```

3. Attach a forward hook and print shapes.

4. Skim `F.linear` in [`functional.py`](../torch/nn/functional.py) (search
   `def linear` / `_add_docstr`).

## First useful PRs

- Fix `extra_repr` / docstring mismatches on a Module.
- Add a missing shape check with a clear error.
- Small util in `nn.utils` with tests.
- Avoid mega-refactors of `module.py` without discussion—it is load-bearing.

## Related files

- [`torch/nn/modules/module.py`](../torch/nn/modules/module.py)
- [`torch/nn/modules/linear.py`](../torch/nn/modules/linear.py)
- [`torch/nn/parameter.py`](../torch/nn/parameter.py)
- [`torch/nn/functional.py`](../torch/nn/functional.py)
- [`torch/nn/init.py`](../torch/nn/init.py)
- [`torch/nn/modules/lazy.py`](../torch/nn/modules/lazy.py)

## Next

[07-optimizers.md](07-optimizers.md) — applying gradients to Parameters.
