# JAX Migration Summary

## Overview

Both `mass2height.py` and `vertical_velocity.py` operators have been successfully rewritten to use JAX for GPU acceleration and improved performance.

## Changes Made

### 1. Mass to Height Operator (`src/operators/mass2height.py`)

**Key Changes:**
- Replaced `numpy` with `jax.numpy`
- Added `@jit` decorators for JIT compilation
- Replaced Python loops with `lax.fori_loop` for better performance
- Used `.at[].set()` syntax for array updates (JAX immutable arrays)
- Added comprehensive docstrings with JAX-specific details

**Functions Updated:**
- `_alpha()` - Alpha coefficient computation
- `dp_faces_p()` - Relative pressure difference computation
- `z_faces()` - Height coordinate on interface levels
- `mass2height_coordinates()` - Main coordinate transformation

**Performance Benefits:**
- JIT compilation for faster execution
- GPU acceleration support
- Automatic differentiation capability
- Vectorized operations throughout

### 2. Vertical Velocity Operator (`src/operators/vertical_velocity.py`)

**Key Changes:**
- Replaced `numpy` with `jax.numpy`
- Added `@jit` decorator for JIT compilation
- Added epsilon (1e-10) to avoid division by zero
- Used JAX's native `gradient()` and array operations
- Improved numerical stability

**Functions Updated:**
- `vertical_divergence_to_vertical_velocity()` - Main velocity computation

**Performance Benefits:**
- JIT compilation for faster execution
- GPU acceleration support
- Automatic differentiation capability
- Better numerical stability with epsilon safeguards

### 3. Test Suite Updates

**test_mass2height.py:**
- Added JAX imports (`jax.numpy as jnp`)
- Added helper functions `to_jax()` and `to_numpy()` for conversions
- All 23 tests remain functional with JAX backend
- Tests automatically convert between NumPy and JAX arrays

**test_vertical_velocity.py:**
- Added JAX imports (`jax.numpy as jnp`)
- Added helper functions `to_jax()` and `to_numpy()` for conversions
- All 13 tests remain functional with JAX backend
- Tests automatically convert between NumPy and JAX arrays

### 4. Dependencies

**Updated `pyproject.toml`:**
```toml
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    ...
]
```

## Migration Benefits

### Performance
- **JIT Compilation**: Functions are compiled on first call, subsequent calls are faster
- **GPU Support**: Transparent GPU acceleration when available
- **Vectorization**: Efficient array operations without explicit loops
- **Memory Efficiency**: Better memory management through functional programming

### Features
- **Auto-differentiation**: Built-in gradient computation capability
- **Automatic Batching**: Easy parallelization with `vmap`
- **Device Flexibility**: Same code runs on CPU, GPU, or TPU
- **Reproducibility**: Explicit random number generation

### Code Quality
- **Immutable Arrays**: Prevents accidental modifications
- **Pure Functions**: Better testability and reliability
- **Type Hints**: Better IDE support with JAX types
- **Documentation**: Improved docstrings with JAX-specific information

## Important JAX Considerations

### Array Updates
JAX arrays are immutable. Instead of:
```python
arr[i] = value  # NumPy style
```
Use:
```python
arr = arr.at[i].set(value)  # JAX style
```

### Loops
Replace Python loops with JAX primitives for better performance:
```python
# Instead of:
for i in range(n):
    arr = update(arr, i)

# Use:
arr = lax.fori_loop(0, n, update_fn, arr)
```

### JIT Compilation
Functions decorated with `@jit` have restrictions:
- No Python side effects
- Array shapes must be static or known at compile time
- Control flow should use JAX primitives (`lax.cond`, `lax.while_loop`, etc.)

### Device Management
JAX automatically selects the best device. To explicitly control:
```python
import jax
jax.devices()  # List available devices
jax.default_device()  # Get default device

# Put arrays on specific devices
arr_gpu = jax.device_put(arr, jax.devices('gpu')[0])
```

## Testing

All existing tests pass with the JAX implementation:

```bash
# Run mass2height tests
pytest tests/test_mass2height.py -v

# Run vertical_velocity tests
pytest tests/test_vertical_velocity.py -v

# Run all tests
pytest tests/ -v
```

## Usage Example

```python
import jax.numpy as jnp
from operators.mass2height import mass2height_coordinates

# Create input arrays (can be NumPy or JAX arrays)
hybrid_coef_A = jnp.array([...])
hybrid_coef_B = jnp.array([...])
surface_pressure = jnp.array([...])
temperature = jnp.array([...])
z_surface = jnp.array([...])

# Call the function (JIT compiled on first call)
zcr = mass2height_coordinates(
    hybrid_coef_A, hybrid_coef_B,
    surface_pressure, temperature, z_surface,
    Rd=287.0, Rd_cpd=287.0/1004.0, gravity0=9.81,
    nx=1013, ny=757, nz=90
)

# Result is a JAX array, convert to NumPy if needed
zcr_numpy = np.array(zcr)
```

## Performance Notes

1. **First Call Overhead**: JIT compilation adds overhead on first call
2. **Warm-up**: Subsequent calls are much faster
3. **GPU Availability**: Check for GPU with `jax.devices('gpu')`
4. **Memory**: JAX may use more memory initially for compilation cache
5. **Batch Processing**: Use `vmap` for batch processing across different inputs

## Backward Compatibility

The API remains identical to the NumPy version:
- Same function signatures
- Same parameter names and types
- Same return types
- NumPy arrays are automatically converted to JAX arrays

This ensures existing code continues to work without modifications.

## Future Improvements

Potential enhancements:
1. Add `vmap` support for batch processing multiple time steps
2. Implement custom gradients for specific operations
3. Add mixed precision support (float16/float32)
4. Optimize memory usage with `jax.checkpoint`
5. Add profiling utilities for performance monitoring

## Installation

To use the JAX-accelerated operators:

### Using uv (recommended)

```bash
# Install with basic dependencies
uv pip install -e .

# Or using uv sync (recommended for development)
uv sync

# For GPU support (NVIDIA)
uv pip install jax[cuda12]

# For TPU support
uv pip install jax[tpu]
```

### Using pip

```bash
# Install with basic dependencies
pip install -e .

# For GPU support (NVIDIA)
pip install jax[cuda12]

# For TPU support
pip install jax[tpu]
```

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
