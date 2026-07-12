# PyTorch Core Systems Onboarding

Personal deep-dive documentation to PyTorch's core tensor, autograd, neural-net, and optimizer stack.

This folder is **not** official Sphinx user documentation. It lives beside the
repo so you can read it in an IDE or on GitHub. Official API docs remain under
[`docs/source/`](../docs/source/); build/test mechanics remain in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Audience

- You know Python and basic PyTorch (`Tensor`, `nn.Module`, `loss.backward()`,
  `optimizer.step()`).
- You are actively learning deep learning (backprop, SGD/Adam intuition).
- You want to read and change the implementation of:

  - [`c10/`](../c10/) — core types, storage, devices, dispatch keys
  - the **dispatcher** — [`aten/src/ATen/core/dispatch/`](../aten/src/ATen/core/dispatch/)
  - [`aten/`](../aten/) — operators and kernels
  - [`torch/autograd/`](../torch/autograd/) + [`torch/csrc/autograd/`](../torch/csrc/autograd/)
  - [`torch/nn/`](../torch/nn/)
  - [`torch/optim/`](../torch/optim/)

Everything else (Dynamo, Inductor, distributed, JIT, quantization, …) is treated
as a black box unless a single sentence is required for context.

## How to use this guide

**Linear path (recommended for months 0–2):** read docs `00` → `08` in order.
Theory comes before code; each page ends with related files and a “next read.”

**Task path:** jump via the index below when you already know what you are
debugging (e.g. “inplace view blew up” → [`13-common-pitfalls.md`](13-common-pitfalls.md)).

**Contribution path:** follow [`14-six-month-roadmap.md`](14-six-month-roadmap.md)
and land PRs while reading—not after finishing every page.

## Document index

| Doc | Topic |
|-----|--------|
| [00-stack-and-mental-model.md](00-stack-and-mental-model.md) | Layered architecture and dependency map |
| [01-theory-primer.md](01-theory-primer.md) | Autograd, layouts, dispatch, backprop theory |
| [02-c10-tensor-and-storage.md](02-c10-tensor-and-storage.md) | `TensorImpl`, `Storage`, dtype, device, views |
| [03-dispatcher.md](03-dispatcher.md) | `DispatchKeySet`, call/redispatch, registration |
| [04-aten-and-codegen.md](04-aten-and-codegen.md) | `native_functions.yaml`, torchgen, kernels |
| [05-autograd-engine.md](05-autograd-engine.md) | Node/Edge/Engine, VariableType, derivatives |
| [06-nn-modules.md](06-nn-modules.md) | `Module`, hooks, `functional` pattern |
| [07-optimizers.md](07-optimizers.md) | Optimizer state, foreach/fused, functional API |
| [08-end-to-end-training-step.md](08-end-to-end-training-step.md) | Full `Linear` + CE + Adam call graph |
| [09-coding-conventions.md](09-coding-conventions.md) | Patterns and style for these trees |
| [10-performance.md](10-performance.md) | Hot paths and common perf levers |
| [11-extension-points.md](11-extension-points.md) | Where to plug in new behavior |
| [12-history-and-design.md](12-history-and-design.md) | Why the code looks the way it does |
| [13-common-pitfalls.md](13-common-pitfalls.md) | Failure modes you will hit |
| [14-six-month-roadmap.md](14-six-month-roadmap.md) | Reading order + contribution milestones |

**Diagram companions:** [`diagrams/dependency-map.md`](diagrams/dependency-map.md),
[`diagrams/call-graphs.md`](diagrams/call-graphs.md).

## Existing in-tree READMEs (read these; do not reinvent)

| Path | Use for |
|------|---------|
| [`aten/src/ATen/native/README.md`](../aten/src/ATen/native/README.md) | Authoring operators |
| [`aten/src/ATen/native/cpu/README.md`](../aten/src/ATen/native/cpu/README.md) | CPU ISA / `DispatchStub` |
| [`aten/src/ATen/core/dispatch/README.md`](../aten/src/ATen/core/dispatch/README.md) | Dispatcher file map |
| [`aten/src/ATen/core/op_registration/README.md`](../aten/src/ATen/core/op_registration/README.md) | Custom op registration |
| [`torch/csrc/autograd/README.md`](../torch/csrc/autograd/README.md) | C++/Python dual types |
| [`c10/core/impl/README.md`](../c10/core/impl/README.md) | `impl/` vs `util/` |
| [`c10/core/impl/README-cow.md`](../c10/core/impl/README-cow.md) | Copy-on-write storage |
| [`c10/core/DispatchKey.h`](../c10/core/DispatchKey.h) | Dispatch key design notes (comments) |

## Diagram conventions

- **Mermaid flowcharts** show ownership or layering (who depends on whom).
- **Mermaid sequence diagrams** show a single op or training step over time.
- **File paths** are repo-root relative (`torch/nn/modules/module.py`).
- Boxes labeled *black box* are out of scope; you may ignore their internals.


## Next

Start here → [00-stack-and-mental-model.md](00-stack-and-mental-model.md).
