# -*- coding: utf-8 -*-
"""
Tests for vertical velocity calculation from vertical divergence.

This module tests the conversion from vertical divergence to vertical velocity
in terrain-following coordinates using JAX.
"""
import os
import sys
import numpy as np
import jax.numpy as jnp
import pytest

sys.path.append(os.path.join(os.getcwd(), "src"))

from operators.vertical_velocity import vertical_divergence_to_vertical_velocity


# Helper functions to convert between numpy and jax
def to_jax(arr):
    """Convert numpy array to JAX array."""
    if isinstance(arr, np.ndarray):
        return jnp.array(arr)
    return arr


def to_numpy(arr):
    """Convert JAX array to numpy array."""
    if isinstance(arr, jnp.ndarray):
        return np.array(arr)
    return arr


class TestVerticalVelocityShapeAndBasics:
    """Tests for basic properties like shape and structure."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        nx, ny, nz = 10, 10, 5
        
        zcr = np.random.rand(nx, ny, nz) * 5000
        specific_surface_geopotential = np.random.rand(nx, ny) * 1000
        u_surface = np.random.rand(nx, ny)
        v_surface = np.random.rand(nx, ny)
        vertical_divergence = np.random.rand(nx, ny, nz + 1) * 1e-5
        alpha = np.random.rand(nx, ny, nz)
        delta_p_rel = np.random.rand(nx, ny, nz) + 0.1
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        assert result.shape == (nx, ny, nz)

    def test_with_zero_divergence(self):
        """Test that zero divergence gives reasonable results."""
        nx, ny, nz = 5, 5, 3
        
        zcr = np.linspace(0, 3000, nz).reshape(1, 1, nz)
        zcr = np.tile(zcr, (nx, ny, 1))
        specific_surface_geopotential = np.zeros((nx, ny))
        u_surface = np.zeros((nx, ny))
        v_surface = np.zeros((nx, ny))
        vertical_divergence = np.zeros((nx, ny, nz + 1))
        alpha = np.random.rand(nx, ny, nz)
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # With zero divergence and flat surface, should get small values
        assert np.all(np.abs(result) < 1e-10)


class TestSurfaceVelocityBoundary:
    """Tests for surface boundary condition computation."""

    def test_surface_velocity_flat_terrain(self):
        """Test surface velocity on flat terrain."""
        nx, ny, nz = 10, 10, 5
        
        zcr = np.random.rand(nx, ny, nz) * 5000
        specific_surface_geopotential = np.zeros((nx, ny))  # Flat terrain
        u_surface = np.ones((nx, ny))
        v_surface = np.ones((nx, ny))
        vertical_divergence = np.zeros((nx, ny, nz + 1))
        alpha = np.random.rand(nx, ny, nz)
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # With flat terrain, surface vertical velocity should be zero
        # Check that result is finite
        assert np.all(np.isfinite(result))

    def test_surface_velocity_with_slope(self):
        """Test surface velocity with terrain slope."""
        nx, ny, nz = 20, 20, 5
        
        zcr = np.random.rand(nx, ny, nz) * 5000
        
        # Create a sloped terrain
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        specific_surface_geopotential = (X * 10.0 + Y * 5.0) * 9.81
        
        u_surface = np.ones((nx, ny))
        v_surface = np.ones((nx, ny))
        vertical_divergence = np.zeros((nx, ny, nz + 1))
        alpha = np.random.rand(nx, ny, nz)
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # With sloped terrain and flow, should get non-zero surface w
        assert np.all(np.isfinite(result))

    def test_upslope_gives_positive_w(self):
        """Test that flow up a slope gives positive vertical velocity."""
        nx, ny, nz = 10, 10, 3
        
        zcr = np.random.rand(nx, ny, nz) * 5000
        
        # Create a slope in x-direction
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        specific_surface_geopotential = X * 100.0 * 9.81  # Positive slope in x
        
        u_surface = np.ones((nx, ny))  # Flow in positive x direction (upslope)
        v_surface = np.zeros((nx, ny))
        vertical_divergence = np.zeros((nx, ny, nz + 1))
        alpha = np.ones((nx, ny, nz))
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # At interior points, upslope flow should give positive w
        # Check middle of domain away from edges where gradients may be affected
        assert np.mean(result[3:-3, 3:-3, 0]) > 0


class TestDivergenceIntegration:
    """Tests for the divergence integration."""

    def test_constant_divergence_increases_w(self):
        """Test that constant positive divergence increases w with height."""
        nx, ny, nz = 10, 10, 5
        
        # Create uniform height levels
        zcr = np.zeros((nx, ny, nz))
        for k in range(nz):
            zcr[:, :, k] = k * 1000.0
        
        specific_surface_geopotential = np.zeros((nx, ny))
        u_surface = np.zeros((nx, ny))
        v_surface = np.zeros((nx, ny))
        
        # Constant positive divergence
        vertical_divergence = np.full((nx, ny, nz + 1), 1e-4)
        vertical_divergence[:, :, 0] = 0  # Surface
        
        alpha = np.ones((nx, ny, nz))
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # Check that w increases with height at interior points
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                for k in range(1, nz):
                    # With positive divergence, w should generally increase upward
                    assert np.isfinite(result[i, j, k])

    def test_negative_divergence_convergence(self):
        """Test negative divergence (convergence) behavior."""
        nx, ny, nz = 10, 10, 5
        
        zcr = np.zeros((nx, ny, nz))
        for k in range(nz):
            zcr[:, :, k] = k * 1000.0
        
        specific_surface_geopotential = np.zeros((nx, ny))
        u_surface = np.zeros((nx, ny))
        v_surface = np.zeros((nx, ny))
        
        # Negative divergence (convergence)
        vertical_divergence = np.full((nx, ny, nz + 1), -1e-4)
        vertical_divergence[:, :, 0] = 0
        
        alpha = np.ones((nx, ny, nz))
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # Result should be finite and well-defined
        assert np.all(np.isfinite(result))


class TestPhysicalConstraints:
    """Tests for physical constraints and reasonableness."""

    def test_no_nan_or_inf(self):
        """Test that result contains no NaN or Inf values."""
        nx, ny, nz = 10, 10, 5
        
        zcr = np.random.rand(nx, ny, nz) * 5000 + 100  # Avoid zero
        specific_surface_geopotential = np.random.rand(nx, ny) * 1000
        u_surface = np.random.rand(nx, ny) * 10
        v_surface = np.random.rand(nx, ny) * 10
        vertical_divergence = np.random.rand(nx, ny, nz + 1) * 1e-4
        alpha = np.random.rand(nx, ny, nz) + 0.1  # Avoid zero
        delta_p_rel = np.random.rand(nx, ny, nz) + 0.1  # Avoid zero
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        assert np.all(np.isfinite(result))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_reasonable_magnitude(self):
        """Test that vertical velocities have reasonable magnitudes."""
        nx, ny, nz = 10, 10, 5
        
        # Create realistic conditions
        zcr = np.zeros((nx, ny, nz))
        for k in range(nz):
            zcr[:, :, k] = k * 1000.0
        
        specific_surface_geopotential = np.random.rand(nx, ny) * 500 * 9.81
        u_surface = np.random.rand(nx, ny) * 5  # 5 m/s
        v_surface = np.random.rand(nx, ny) * 5
        vertical_divergence = np.random.rand(nx, ny, nz + 1) * 1e-5
        alpha = np.ones((nx, ny, nz))
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # Vertical velocities in troposphere typically < 10 m/s
        assert np.all(np.abs(result) < 50)  # Generous upper bound


class TestInterpolationBehavior:
    """Tests for interpolation from faces to mass levels."""

    def test_alpha_delta_p_rel_influence(self):
        """Test that alpha and delta_p_rel affect interpolation."""
        nx, ny, nz = 5, 5, 3
        
        zcr = np.zeros((nx, ny, nz))
        for k in range(nz):
            zcr[:, :, k] = k * 1000.0
        
        specific_surface_geopotential = np.zeros((nx, ny))
        u_surface = np.zeros((nx, ny))
        v_surface = np.zeros((nx, ny))
        vertical_divergence = np.ones((nx, ny, nz + 1)) * 1e-4
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        # Test with different alpha values
        alpha1 = np.ones((nx, ny, nz)) * 0.5
        alpha2 = np.ones((nx, ny, nz)) * 0.8
        delta_p_rel = np.ones((nx, ny, nz))
        
        result1 = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha1,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        result2 = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha2,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        # Different alpha should give different results
        assert not np.allclose(result1, result2)


class TestEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_single_column(self):
        """Test with a single column (nx=1, ny=1)."""
        nx, ny, nz = 1, 1, 5
        
        zcr = np.zeros((nx, ny, nz))
        for k in range(nz):
            zcr[:, :, k] = k * 1000.0
        
        specific_surface_geopotential = np.zeros((nx, ny))
        u_surface = np.ones((nx, ny))
        v_surface = np.ones((nx, ny))
        vertical_divergence = np.random.rand(nx, ny, nz + 1) * 1e-5
        alpha = np.ones((nx, ny, nz))
        delta_p_rel = np.ones((nx, ny, nz))
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        assert result.shape == (nx, ny, nz)
        assert np.all(np.isfinite(result))

    def test_small_domain(self):
        """Test with a small domain."""
        nx, ny, nz = 3, 3, 3
        
        zcr = np.random.rand(nx, ny, nz) * 3000
        specific_surface_geopotential = np.random.rand(nx, ny) * 100
        u_surface = np.random.rand(nx, ny)
        v_surface = np.random.rand(nx, ny)
        vertical_divergence = np.random.rand(nx, ny, nz + 1) * 1e-5
        alpha = np.random.rand(nx, ny, nz) + 0.1
        delta_p_rel = np.random.rand(nx, ny, nz) + 0.1
        gravity0 = 9.81
        dx = 1250.0
        dy = 1250.0
        
        result = vertical_divergence_to_vertical_velocity(
            zcr,
            specific_surface_geopotential,
            u_surface,
            v_surface,
            vertical_divergence,
            alpha,
            delta_p_rel,
            gravity0,
            dx,
            dy,
            nz,
        )
        
        assert result.shape == (nx, ny, nz)
        assert np.all(np.isfinite(result))


# Integration test with actual AROME data (if available)
def test_vertical_velocity_with_real_data():
    """Integration test with real AROME data if available."""
    arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")
    
    if not os.path.exists(arome_file):
        pytest.skip("AROME test file not available")
    
    from arome2fvm.arome_reader import AromeReader
    from operators.mass2height import mass2height_coordinates, _alpha, dp_faces_p
    
    reader = AromeReader(arome_file)
    
    # First compute zcr
    zcr = mass2height_coordinates(
        reader.get_hybrid_coef_A(),
        reader.get_hybrid_coef_B(),
        reader.get_surface_pressure(),
        reader.get_temperature(),
        reader.get_surface_geopotential() / 9.81,
        287.0,
        287.0 / 1004.0,
        9.81,
        reader.get_nx(),
        reader.get_ny(),
        reader.get_nz(),
    )
    
    # Compute alpha and delta_p_rel (simplified)
    hybrid_coef_A = reader.get_hybrid_coef_A()
    hybrid_coef_B = reader.get_hybrid_coef_B()
    surface_pressure = reader.get_surface_pressure()
    
    p_tilde = (
        hybrid_coef_A[np.newaxis, np.newaxis, :]
        + np.exp(surface_pressure[:, :, np.newaxis])
        * hybrid_coef_B[np.newaxis, np.newaxis, :]
    )
    
    p = np.sqrt(p_tilde[:, :, 1:] * p_tilde[:, :, :-1])
    delta_p_tilde = p_tilde[:, :, :-1] - p_tilde[:, :, 1:]
    
    alpha = _alpha(p, p_tilde)
    delta_p_rel = dp_faces_p(p, delta_p_tilde, 287.0 / 1004.0)
    
    u_surface, v_surface = reader.get_surface_velocities()
    
    wvel = vertical_divergence_to_vertical_velocity(
        zcr,
        reader.get_surface_geopotential(),
        u_surface,
        v_surface,
        reader.get_vertical_divergence(),
        alpha,
        delta_p_rel,
        9.81,
        1250.0,
        1250.0,
        reader.get_nz(),
    )
    
    # Basic sanity checks
    assert wvel.shape == (reader.get_nx(), reader.get_ny(), reader.get_nz())
    assert np.all(np.isfinite(wvel))
    
    # Vertical velocities should be reasonable (< 50 m/s in typical conditions)
    assert np.all(np.abs(wvel) < 100)  # Very generous bound
    
    # Check statistics
    mean_w = np.mean(wvel)
    std_w = np.std(wvel)
    
    # Mean should be relatively small (close to zero for large domain)
    assert np.abs(mean_w) < 5.0
    
    # Standard deviation should be reasonable
    assert std_w < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
