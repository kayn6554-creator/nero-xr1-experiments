"""Runtime compatibility for the CANN 8.0 / PyTorch 2.1 baseline."""

from __future__ import annotations

import inspect

import torch.utils._pytree as pytree


def patch_torch_pytree() -> None:
    """Expose the public pytree API expected by Transformers 4.57.

    PyTorch 2.1 only exposes ``_register_pytree_node``.  Newer Transformers
    calls the public spelling and supplies metadata keywords introduced after
    2.1.  The metadata is optional for model execution, so pass through only
    keywords understood by the installed private implementation.
    """
    if hasattr(pytree, "register_pytree_node"):
        return

    private = pytree._register_pytree_node
    supported = set(inspect.signature(private).parameters)

    def register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs):
        compatible = {key: value for key, value in kwargs.items() if key in supported}
        return private(typ, flatten_fn, unflatten_fn, **compatible)

    pytree.register_pytree_node = register_pytree_node


patch_torch_pytree()