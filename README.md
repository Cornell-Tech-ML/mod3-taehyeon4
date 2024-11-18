# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Parallel diagnostic output

````
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (163)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # Check if tensors are stride-aligned                                |
        is_aligned = (                                                       |
            len(out_shape) == len(in_shape)                                  |
            and len(out_strides) == len(in_strides)                          |
            and (out_shape == in_shape).all()--------------------------------| #0
            and (out_strides == in_strides).all()----------------------------| #1
        )                                                                    |
                                                                             |
        if is_aligned:                                                       |
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #3
                out_index = np.empty(MAX_DIMS, dtype=np.int32)               |
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                                                                             |
                in_pos = index_to_position(in_index, in_strides)             |
                out_pos = index_to_position(out_index, out_strides)          |
                out[out_pos] = fn(in_storage[in_pos])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (184) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (185) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (219)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (219)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # Check if tensors are stride-aligned                              |
        is_aligned = (                                                     |
            len(out_strides) == len(a_strides)                             |
            and len(out_shape) == len(a_shape)                             |
            and len(a_strides) == len(b_strides)                           |
            and len(b_shape) == len(a_shape)                               |
            and (out_strides == a_strides).all()---------------------------| #4
            and (a_strides == b_strides).all()-----------------------------| #5
            and (out_shape == a_shape).all()-------------------------------| #6
            and (a_shape == b_shape).all()---------------------------------| #7
        )                                                                  |
                                                                           |
        if is_aligned:                                                     |
            for i in prange(len(out)):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    |
        else:                                                              |
            for i in prange(len(out)):-------------------------------------| #9
                # Create index buffers                                     |
                out_index = np.empty(MAX_DIMS, dtype=np.int32)             |
                a_index = np.empty(MAX_DIMS, dtype=np.int32)               |
                b_index = np.empty(MAX_DIMS, dtype=np.int32)               |
                                                                           |
                to_index(i, out_shape, out_index)                          |
                                                                           |
                # Handle broadcasting                                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                           |
                # Calculate positions                                      |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                out_pos = index_to_position(out_index, out_strides)        |
                                                                           |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (248) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (249) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (250) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (289)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (289)
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   |
        out: Storage,                                              |
        out_shape: Shape,                                          |
        out_strides: Strides,                                      |
        a_storage: Storage,                                        |
        a_shape: Shape,                                            |
        a_strides: Strides,                                        |
        reduce_dim: int,                                           |
    ) -> None:                                                     |
        for i in prange(len(out)):---------------------------------| #10
            out_index = np.empty(MAX_DIMS, dtype=np.int32)         |
            reduce_size = a_shape[reduce_dim]                      |
            to_index(i, out_shape, out_index)                      |
            out_pos = index_to_position(out_index, out_strides)    |
                                                                   |
            reduce_stride = a_strides[reduce_dim]                  |
            base_pos = index_to_position(out_index, a_strides)     |
            acc = out[out_pos]                                     |
                                                                   |
            for j in range(reduce_size):                           |
                a_pos = base_pos + j * reduce_stride               |
                acc = fn(acc, float(a_storage[a_pos]))             |
                                                                   |
            out[out_pos] = acc                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (299) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (317)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/taehyeon/repositories/mod3-taehyeon4/minitorch/fast_ops.py (317)
--------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                    |
    out: Storage,                                                               |
    out_shape: Shape,                                                           |
    out_strides: Strides,                                                       |
    a_storage: Storage,                                                         |
    a_shape: Shape,                                                             |
    a_strides: Strides,                                                         |
    b_storage: Storage,                                                         |
    b_shape: Shape,                                                             |
    b_strides: Strides,                                                         |
) -> None:                                                                      |
    """NUMBA tensor matrix multiply function.                                   |
                                                                                |
    Should work for any tensor shapes that broadcast as long as                 |
                                                                                |
    ```                                                                         |
    assert a_shape[-1] == b_shape[-2]                                           |
    ```                                                                         |
                                                                                |
    Optimizations:                                                              |
                                                                                |
    * Outer loop in parallel                                                    |
    * No index buffers or function calls                                        |
    * Inner loop should have no global writes, 1 multiply.                      |
                                                                                |
                                                                                |
    Args:                                                                       |
    ----                                                                        |
        out (Storage): storage for `out` tensor                                 |
        out_shape (Shape): shape for `out` tensor                               |
        out_strides (Strides): strides for `out` tensor                         |
        a_storage (Storage): storage for `a` tensor                             |
        a_shape (Shape): shape for `a` tensor                                   |
        a_strides (Strides): strides for `a` tensor                             |
        b_storage (Storage): storage for `b` tensor                             |
        b_shape (Shape): shape for `b` tensor                                   |
        b_strides (Strides): strides for `b` tensor                             |
                                                                                |
    Returns:                                                                    |
    -------                                                                     |
        None : Fills in `out`                                                   |
                                                                                |
    """                                                                         |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                      |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                      |
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0                |
                                                                                |
    for batch in prange(out_shape[0]):------------------------------------------| #11
        a_batch_idx = batch * a_batch_stride                                    |
        b_batch_idx = batch * b_batch_stride                                    |
        out_batch_idx = batch * out_batch_stride                                |
                                                                                |
        for r in range(out_shape[-2]):                                          |
            a_r_stride = r * a_strides[-2]                                      |
            out_r_stride = r * out_strides[-2]                                  |
            for c in range(out_shape[-1]):                                      |
                b_c_stride = c * b_strides[-1]                                  |
                out_idx = out_batch_idx + out_r_stride + c * out_strides[-1]    |
                acc = 0.0                                                       |
                for k in range(a_shape[-1]):                                    |
                    a_idx = a_batch_idx + a_r_stride + k * a_strides[-1]        |
                    b_idx = b_batch_idx + k * b_strides[-2] + b_c_stride        |
                    acc += a_storage[a_idx] * b_storage[b_idx]                  |
                out[out_idx] = acc                                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
````
