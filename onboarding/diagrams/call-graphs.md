# Diagrams — Call Graphs

Companion to [08-end-to-end-training-step.md](../08-end-to-end-training-step.md)
and [03-dispatcher.md](../03-dispatcher.md).

## Op call: `torch.add` / `x + y`

```mermaid
sequenceDiagram
  participant Py as Python x+y
  participant Bind as generated binding
  participant At as at::add
  participant Disp as Dispatcher
  participant AG as VariableType::add
  participant Native as at::native::add
  participant Node as AddBackward0
  Py->>Bind: THP unwrap
  Bind->>At: at::add tensors
  At->>Disp: call
  Disp->>AG: AutogradCPU kernel
  Note over AG: AutoDispatchBelowADInplaceOrView
  AG->>Disp: redispatch
  Disp->>Native: CPU or CUDA
  Native-->>AG: result
  AG->>Node: create if requires_grad
  AG-->>Py: Tensor with grad_fn
```

## Module forward: `Linear`

```mermaid
sequenceDiagram
  participant U as User
  participant Call as Module._call_impl
  participant Fwd as Linear.forward
  participant F as F.linear
  participant Stack as Dispatcher+Autograd+ATen
  U->>Call: model(x)
  alt hooks present
    Call->>Call: forward_pre_hooks
    Call->>Fwd: forward
    Call->>Call: forward_hooks
  else no hooks
    Call->>Fwd: fast path forward
  end
  Fwd->>F: F.linear x W b
  F->>Stack: at::linear / addmm
  Stack-->>U: logits
```

## Backward: `loss.backward()`

```mermaid
sequenceDiagram
  participant U as User
  participant PE as PythonEngine
  participant E as Engine
  participant RQ as ReadyQueue
  participant N as Node.apply
  participant Acc as AccumulateGrad
  U->>PE: backward
  PE->>E: execute
  E->>E: GraphRoot + dependencies
  loop until empty
    E->>RQ: pop NodeTask
    RQ->>N: apply InputBuffer
    N->>RQ: schedule consumers
  end
  N->>Acc: leaf grads
  Acc-->>U: param.grad
```

## Optimizer: `Adam.step`

```mermaid
sequenceDiagram
  participant U as User
  participant Adam as Adam.step
  participant Fn as adam functional
  participant Impl as single or foreach or fused
  participant ATen as ATen kernels
  U->>Adam: step
  Adam->>Adam: collect params grads state
  Adam->>Fn: adam lists
  Fn->>Impl: select tier
  Impl->>ATen: _foreach_* or _fused_adam_
  ATen-->>U: params updated inplace
```

## Full training iteration

```mermaid
flowchart TD
  z["opt.zero_grad"]
  f["model forward Module F dispatcher Autograd kernels"]
  loss["loss op builds more Nodes"]
  b["loss.backward Engine VJPs"]
  g["param.grad filled"]
  s["opt.step foreach or fused"]
  z --> f --> loss --> b --> g --> s
```

## File anchors per hop

| Hop | Primary files |
|-----|----------------|
| Module call | `torch/nn/modules/module.py` |
| Linear | `torch/nn/modules/linear.py` |
| Functional | `torch/nn/functional.py` |
| Bindings | `torch/csrc/autograd/generated/python_*.cpp` (build) |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.h` |
| Autograd wrap | `tools/autograd/gen_variable_type.py` → generated VariableType |
| Engine | `torch/csrc/autograd/engine.cpp` |
| AccumulateGrad | `torch/csrc/autograd/functions/` |
| Adam | `torch/optim/adam.py` |
| Fused kernel | `aten/src/ATen/native/cuda/FusedAdam*.cu` |

## Related pages

- [05-autograd-engine.md](../05-autograd-engine.md)
- [06-nn-modules.md](../06-nn-modules.md)
- [07-optimizers.md](../07-optimizers.md)
- [08-end-to-end-training-step.md](../08-end-to-end-training-step.md)
