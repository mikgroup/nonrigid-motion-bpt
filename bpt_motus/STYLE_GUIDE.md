# BPT-MOTUS Style Guide

## Overall Philosophy

Classes should have a clear lifecycle:

1. Constructor (`__init__`) defines object state.
2. Public methods execute workflow steps.
3. Helper methods implement internal details.

Object construction should be lightweight and predictable.

Code may be reorganized to improve consistency and readability as long as functionality is preserved.

---

## Constructor (`__init__`)

The constructor should:

* Store input arguments as class attributes.
* Define all input and output filenames as `*_fname` attributes.
* Declare important attributes that will be populated later.
* Initialize those attributes to `None` or an empty container (e.g., `[]`).
* Include type annotations whenever practical.
* Construct lightweight helper objects if required.
* Resolve the compute device dynamically if not provided (e.g., `self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")`).

### Important Attributes

Important attributes include:

* Data loaded from disk
* Data saved to disk
* Outputs returned to users
* Intermediate state shared between methods
* Cached computations
* Models and operators used across methods

Do not declare temporary loop variables, scratch variables, or short-lived intermediates.

### Attribute Typing

Important class attributes should be declared with type annotations when monitoring if doing so is necessary to improve consistency and behavior is preserved.

---

## Refactoring Rules

The following changes are allowed:

* Reorganizing method order
* Creating helper methods
* Moving loading logic out of `__init__`
* Moving preprocessing out of `__init__`
* Converting local filenames into `self.*_fname`
* Declaring important attributes in `__init__`
* Adding type annotations
* Improving docstrings
* Framework type promotion (e.g., converting NumPy arrays to PyTorch Tensors) during data loading or field construction stages to optimize runtime loops
* Adding new class attributes to store intermediate states, models, or cached computations shared across methods

The following changes are NOT allowed:

* Changing algorithms
* Changing numerical operations (except framework type promotion for performance acceleration)
* Changing tensor shapes
* Changing optimization behavior
* Changing saved outputs
* Renaming public APIs without explicit instruction

---

## Filenames

Input and output files should generally be stored as attributes.

Examples:

```python
self.xk_fname: str
self.coords_fname: str
self.motion_params_fname: str
self.recon_fname: str
```
Avoid local filename variables when the filename is part of the object's persistent state.

---

## Dynamic File Resolution

File and directory paths should be dynamically resolved based on previous class parameters (such as arguments passed into __init__ like cropping factors, processing modes, input directories, etc.) to enforce strict naming conventions and eliminate manual string entry.

Do not hardcode these dynamic parts; instead, use the parameters directly to build the paths.

Example:

```python
def __init__(self, inp_dir: str, crop_factor: int = 3):
    self.inp_dir: str = inp_dir
    
    self.save_dir: str = os.path.join(self.inp_dir, f"crop_{crop_factor}")
    self.S_fname: str = os.path.join(self.save_dir, "S_reference.npy")
```

---

## Docstrings

Every public method and helper method should have a docstring. (Exception: Do not write a docstring for __init__. Use inline comments as specified above.)

Format:

Brief description.

Args:
arg_name (type): Description. (Shape: (...))

Stores:
attr_name (type): Description. (Shape: (...))

Returns:
value_name (type): Description. (Shape: (...))

Only include shape information when known.

If a method neither stores nor returns anything, omit the unused section.

---

## Example File

Use RadialArchive as the primary style reference.

When uncertain, prefer the structure and lifecycle used in RadialArchive.