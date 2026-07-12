# 13 — Common Pitfalls

Failure modes you will hit while contributing to these systems. Each entry:
symptom → cause → fix.

---

## Autograd and tensors

### Inplace modification of a view needed for backward

**Symptom:** `RuntimeError: … modified by an inplace operation`

**Cause:** Saved tensor / view version mismatch after inplace write.

**Fix:** Use out-of-place ops; clone when you must mutate; understand
ADInplaceOrView version bumps ([05-autograd-engine.md](05-autograd-engine.md)).

### Calling `.backward()` twice without `retain_graph=True`

**Symptom:** Graph freed / trying to backward through freed graph.

**Cause:** Nodes freed after first backward by default.

**Fix:** `retain_graph=True` if intentional; usually restructure to one backward.

### Expecting `.grad` on non-leaves

**Symptom:** `tensor.grad is None` on intermediate activations.

**Cause:** Only leaves accumulate `.grad` by default.

**Fix:** `retain_grad()` on intermediates, or use `autograd.grad(...)`.

### `requires_grad=False` leaf surprise

**Symptom:** No graph despite math involving Parameters.

**Cause:** Parameter frozen or tensor created without grad; or ops under
`no_grad`.

**Fix:** Check `param.requires_grad`, GradMode, and whether you detached.

### Detach / numpy / `item()` silently cutting graphs

**Symptom:** Gradients always `None` for “connected” code.

**Cause:** `.detach()`, `.numpy()`, Python scalars from `.item()` break tracking.

**Fix:** Keep computation on tensors until loss is formed.

---

## `nn.Module`

### Assigning a plain Tensor instead of `Parameter`

**Symptom:** Missing from `parameters()` / `state_dict` / optimizer.

**Cause:** `__setattr__` only auto-registers `Parameter` / `Buffer` / `Module`.

**Fix:** `self.w = nn.Parameter(...)` or `register_parameter`.

### Storing Modules in a Python list

**Symptom:** Submodules not moved with `.to()`, not in `state_dict`.

**Cause:** Lists are not `_modules`.

**Fix:** `ModuleList` / `Sequential` / `ModuleDict`.

### Forgetting `super().__init__()`

**Symptom:** Cryptic attribute / registration errors.

**Fix:** Always call `Module.__init__` first.

### Hooks making everything slow

**Symptom:** Unexpected overhead on tiny ops.

**Cause:** Any hooks disable `_call_impl` fast path.

**Fix:** Remove global hooks; scope instrumentation.

---

## Dispatcher / ATen

### Registered kernel “never runs”

**Symptom:** Wrong backend or Autograd fallback behavior.

**Cause:** Registered on wrong DispatchKey (e.g. backend instead of Autograd*
override); or higher-priority key shadows you.

**Fix:** Read key priority notes; use `Autograd` alias correctly
([03-dispatcher.md](03-dispatcher.md), [12-history-and-design.md](12-history-and-design.md)).

### Missing `derivatives.yaml` for Explicit kernel

**Symptom:** Autograd not implemented / wrong grads / fallback errors.

**Cause:** CompositeExplicit or backend kernel without formula.

**Fix:** Add formula or implement `autograd.Function`; or make Implicit
decomposition.

### Wrong alias annotations

**Symptom:** Functionalization / view / inplace assert failures; compiler
breakage.

**Cause:** `Tensor(a!)` vs view annotations incorrect.

**Fix:** Copy annotations from closest sibling op; read native README.

### Confusing `AT_DISPATCH_*` with DispatchKey

**Symptom:** “I dispatched to float” but CUDA kernel still not selected.

**Cause:** Different mechanisms ([04-aten-and-codegen.md](04-aten-and-codegen.md)).

### Editing generated sources

**Symptom:** Changes vanish on rebuild.

**Fix:** Edit yaml / native / derivatives; regenerate via normal build.

---

## Optimizers

### Replacing parameter Tensor objects in `step`

**Symptom:** State loss, Module still holds old tensor, silent training bugs.

**Cause:** `state` and Module point at **identity**; new Tensor breaks links.

**Fix:** Inplace update storage (`param.add_`, fused kernels, foreach).

### Two optimizers / param groups double-counting

**Symptom:** Exploding updates.

**Cause:** Same Parameter in multiple groups or multiple Optimizers.

**Fix:** Ensure disjoint param sets unless intentional and understood.

### Loading `state_dict` after changing model structure

**Symptom:** Size mismatch / missing state.

**Cause:** State keyed/ordered for previous shapes.

**Fix:** Rebuild optimizer or carefully remap state.

### Expecting grads after `zero_grad` before backward

**Symptom:** `step` no-ops.

**Cause:** Order should be forward → backward → step → zero_grad (or zero at
start of iteration).

---

## Memory / layout

### Accidental contiguous copies in a loop

**Symptom:** Host/device bandwidth bound for “simple” math.

**Cause:** Non-contiguous views fed to ops that materialize.

**Fix:** Fix layout once; avoid repeated transpose without planning.

### Holding references to huge saved graphs

**Symptom:** OOM after many iterations.

**Cause:** Retaining outputs / `retain_graph` / storing forward tensors.

**Fix:** Let graphs free; clear references; checkpoint if needed.

---

## Debugging checklist

1. `print(tensor.grad_fn, tensor.requires_grad, tensor.is_leaf)`
2. `print(module._parameters.keys())`
3. Minimal repro under `anomaly_mode` if NaNs appear
4. `rg` the schema in `native_functions.yaml` + matching `derivatives.yaml`
5. Confirm DispatchKey expectations before blaming the numeric kernel

## Related files

- [`torch/nn/modules/module.py`](../torch/nn/modules/module.py)
- [`torch/csrc/autograd/variable.h`](../torch/csrc/autograd/variable.h)
- [`aten/src/ATen/native/README.md`](../aten/src/ATen/native/README.md)
- [`torch/optim/optimizer.py`](../torch/optim/optimizer.py)

## Next

[14-six-month-roadmap.md](14-six-month-roadmap.md).
