# Diagrams — Dependency Map

Companion to [00-stack-and-mental-model.md](../00-stack-and-mental-model.md).

## Full stack dependencies

```mermaid
flowchart TB
  subgraph user [User code]
    train["Training loop"]
  end
  subgraph nn_pkg [torch.nn]
    mod["Module / Parameter / Buffer"]
    fun["functional F.*"]
  end
  subgraph optim_pkg [torch.optim]
    opt["Optimizer / Adam SGD"]
    func_opt["functional adam sgd"]
  end
  subgraph py_bind [Python bindings]
    bind["generated python_*.cpp"]
  end
  subgraph ag [Autograd]
    vt["VariableType wrappers"]
    nodes["Node Edge SavedVariable"]
    eng["Engine GraphTask"]
    deriv["derivatives.yaml codegen"]
  end
  subgraph disp [Dispatcher]
    dcall["Dispatcher::call"]
    dred["redispatch + TLS exclude"]
    keys["DispatchKeySet"]
  end
  subgraph aten_pkg [ATen]
    yaml["native_functions.yaml"]
    native["native kernels TensorIterator"]
    stubs["DispatchStub CPU ISA"]
  end
  subgraph c10_pkg [c10]
    ti["TensorImpl"]
    st["Storage DataPtr Allocator"]
    dk["DispatchKey Device ScalarType"]
  end
  train --> mod
  train --> opt
  mod --> fun
  fun --> bind
  opt --> func_opt
  func_opt --> bind
  bind --> vt
  vt --> nodes
  vt --> dcall
  eng --> nodes
  train -.->|"loss.backward"| eng
  dcall --> dred
  dred --> native
  keys --> dcall
  yaml --> native
  native --> stubs
  native --> ti
  vt --> ti
  ti --> st
  ti --> dk
  deriv --> vt
  deriv --> nodes
```

## Data ownership only

```mermaid
flowchart LR
  Tensor["at::Tensor / torch.Tensor"]
  Impl["TensorImpl"]
  Storage["StorageImpl"]
  Bytes["DataPtr bytes"]
  Meta["AutogradMeta?"]
  Keyset["DispatchKeySet"]
  Tensor --> Impl
  Impl --> Storage
  Storage --> Bytes
  Impl --> Meta
  Impl --> Keyset
```

## Build-time vs run-time

```mermaid
flowchart TB
  subgraph build [Build time]
    nf["native_functions.yaml"]
    dy["derivatives.yaml"]
    tg["torchgen + tools/autograd"]
    gen["generated registrations VariableType Nodes bindings"]
    nf --> tg
    dy --> tg
    tg --> gen
  end
  subgraph run [Run time]
    call["op call"]
    table["OperatorEntry kernel table"]
    kern["native kernel"]
    call --> table --> kern
  end
  gen -->|"links into"| table
```

## Related pages

- [02-c10-tensor-and-storage.md](../02-c10-tensor-and-storage.md)
- [03-dispatcher.md](../03-dispatcher.md)
- [04-aten-and-codegen.md](../04-aten-and-codegen.md)
- [05-autograd-engine.md](../05-autograd-engine.md)
