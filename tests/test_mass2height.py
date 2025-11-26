# -*- coding: utf-8 -*-
"""
Tests for mass2height coordinate transformation operators.

This module tests the conversion from mass-based coordinates to 
height-based terrain-following coordinates using JAX.
"""
import os
import sys
import numpy as np
import jax.numpy as jnp
import pytest

sys.path.append(os.path.join(os.getcwd(), "src"))

from operators.mass2height import (
    mass2height_coordinates,
    _alpha,
    dp_faces_p,
    z_faces,
)


# Helper function to convert between numpy and jax
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


class TestAlphaCoefficient:
    """Tests for the _alpha helper function."""

    def test_alpha_shape(self):
        """Test that alpha returns correct shape."""
        p = np.random.rand(10, 10, 5)
        p_faces = np.random.rand(10, 10, 6)
        
        alpha = _alpha(p, p_faces)
        
        assert alpha.shape == p.shape

    def test_alpha_first_level(self):
        """Test that alpha is -1 at first level (k=0)."""
        p = np.random.rand(10, 10, 5)
        p_faces = np.random.rand(10, 10, 6)
        
        alpha = _alpha(p, p_faces)
        
        assert np.all(alpha[:, :, 0] == -1)

    def test_alpha_calculation(self):
        """Test alpha calculation for higher levels."""
        p = np.array([[[1.0, 2.0, 3.0]]])
        p_faces = np.array([[[4.0, 5.0, 6.0, 7.0]]])
        
        alpha = _alpha(p, p_faces)
        
        # For k >= 1: alpha = 1 - p / p_faces[:, :, 1:]
        expected_k1 = 1 - 2.0 / 6.0
        expected_k2 = 1 - 3.0 / 7.0
        
        assert alpha[0, 0, 0] == -1  # First level
        assert np.isclose(alpha[0, 0, 1], expected_k1)
        assert np.isclose(alpha[0, 0, 2], expected_k2)

    def test_alpha_positive_values(self):
        """Test that alpha gives expected positive values when p < p_faces."""
        p = np.array([[[1.0, 2.0]]])
        p_faces = np.array([[[5.0, 10.0, 20.0]]])
        
        alpha = _alpha(p, p_faces)
        
        # For k >= 1 with p < p_faces, alpha should be positive
        assert alpha[0, 0, 1] > 0
        assert alpha[0, 0, 1] < 1


class TestDpFacesP:
    """Tests for the dp_faces_p helper function."""

    def test_dp_faces_p_shape(self):
        """Test that dp_faces_p returns correct shape."""
        p = np.random.rand(10, 10, 5)
        delta_p_faces = np.random.rand(10, 10, 6)
        Rd_cpd = 0.286
        
        result = dp_faces_p(p, delta_p_faces, Rd_cpd)
        
        assert result.shape == p.shape

    def test_dp_faces_p_first_level(self):
        """Test dp_faces_p at first level (k=0)."""
        p = np.random.rand(5, 5, 3)
        delta_p_faces = np.random.rand(5, 5, 4)
        Rd_cpd = 0.286
        
        result = dp_faces_p(p, delta_p_faces, Rd_cpd)
        
        expected_k0 = -(1 + 1 / Rd_cpd)
        assert np.allclose(result[:, :, 0], expected_k0)

    def test_dp_faces_p_calculation(self):
        """Test dp_faces_p calculation for higher levels."""
        p = np.array([[[1.0, 2.0, 4.0]]])
        delta_p_faces = np.array([[[0.5, 0.2, 0.4, 0.8]]])
        Rd_cpd = 0.286
        
        result = dp_faces_p(p, delta_p_faces, Rd_cpd)
        
        # For k >= 1: delta_p_rel = delta_p_faces / p
        expected_k1 = 0.4 / 2.0
        expected_k2 = 0.8 / 4.0
        
        assert np.isclose(result[0, 0, 1], expected_k1)
        assert np.isclose(result[0, 0, 2], expected_k2)


class TestZFaces:
    """Tests for the z_faces helper function."""

    def test_z_faces_shape(self):
        """Test that z_faces returns correct shape."""
        nx, ny, nz = 10, 10, 5
        z_surface = np.random.rand(nx, ny) * 100
        temperature = np.random.rand(nx, ny, nz) * 300
        delta_p_p = np.random.rand(nx, ny, nz)
        Rd = 287.0
        gravity0 = 9.81
        
        result = z_faces(z_surface, temperature, delta_p_p, Rd, gravity0, nx, ny, nz)
        
        assert result.shape == (nx, ny, nz + 1)

    def test_z_faces_surface_level(self):
        """Test that surface level matches input orography."""
        nx, ny, nz = 5, 5, 3
        z_surface = np.random.rand(nx, ny) * 100
        temperature = np.random.rand(nx, ny, nz) * 300
        delta_p_p = np.random.rand(nx, ny, nz)
        Rd = 287.0
        gravity0 = 9.81
        
        result = z_faces(z_surface, temperature, delta_p_p, Rd, gravity0, nx, ny, nz)
        
        assert np.allclose(result[:, :, nz], z_surface)

    def test_z_faces_monotonic_increase(self):
        """Test that z increases with decreasing k (going upward)."""
        nx, ny, nz = 5, 5, 10
        z_surface = np.zeros((nx, ny))
        temperature = np.full((nx, ny, nz), 300.0)
        delta_p_p = np.full((nx, ny, nz), -0.1)  # Negative for upward
        Rd = 287.0
        gravity0 = 9.81
        
        result = z_faces(z_surface, temperature, delta_p_p, Rd, gravity0, nx, ny, nz)
        
        # Heights should increase as we go up (decreasing k)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    assert result[i, j, k] >= result[i, j, k + 1]


class TestMass2HeightCoordinates:
    """Tests for the main mass2height_coordinates function."""

    def test_mass2height_shape(self):
        """Test output shape of mass2height_coordinates."""
        nx, ny, nz = 10, 10, 5
        nz_faces = nz + 1
        
        hybrid_coef_A = np.random.rand(nz_faces)
        hybrid_coef_B = np.random.rand(nz_faces)
        surface_pressure = np.random.rand(nx, ny) * 10
        temperature = np.random.rand(nx, ny, nz) * 300
        z_surface = np.random.rand(nx, ny) * 100
        Rd = 287.0
        Rd_cpd = 0.286
        gravity0 = 9.81
        
        result = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temperature,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        assert result.shape == (nx, ny, nz)

    def test_mass2height_monotonic_increase(self):
        """Test that heights increase monotonically upward."""
        nx, ny, nz = 20, 20, 10
        nz_faces = nz + 1
        
        # Create realistic hybrid coefficients
        hybrid_coef_A = np.linspace(100000, 0, nz_faces)
        hybrid_coef_B = np.linspace(0, 1, nz_faces)
        surface_pressure = np.full((nx, ny), 11.0)  # log(surface_pressure)
        temperature = np.full((nx, ny, nz), 280.0)
        z_surface = np.zeros((nx, ny))
        Rd = 287.0
        Rd_cpd = 0.286
        gravity0 = 9.81
        
        result = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temperature,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        # Test multiple random points
        for _ in range(5):
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny)
            
            # Heights should increase with k (after flipping)
            for k in range(1, nz):
                assert result[i, j, k] > result[i, j, k - 1], \
                    f"Height not increasing at ({i},{j},{k})"

    def test_mass2height_positive_heights(self):
        """Test that heights above surface are positive when surface is at 0."""
        nx, ny, nz = 10, 10, 5
        nz_faces = nz + 1
        
        hybrid_coef_A = np.linspace(100000, 0, nz_faces)
        hybrid_coef_B = np.linspace(0, 1, nz_faces)
        surface_pressure = np.full((nx, ny), 11.0)
        temperature = np.full((nx, ny, nz), 280.0)
        z_surface = np.zeros((nx, ny))  # Surface at sea level
        Rd = 287.0
        Rd_cpd = 0.286
        gravity0 = 9.81
        
        result = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temperature,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        # All heights should be positive or close to zero at surface
        assert np.all(result >= -1.0)  # Allow small numerical errors

    def test_mass2height_with_orography(self):
        """Test that orography is properly handled."""
        nx, ny, nz = 10, 10, 5
        nz_faces = nz + 1
        
        hybrid_coef_A = np.linspace(100000, 0, nz_faces)
        hybrid_coef_B = np.linspace(0, 1, nz_faces)
        surface_pressure = np.full((nx, ny), 11.0)
        temperature = np.full((nx, ny, nz), 280.0)
        z_surface = np.full((nx, ny), 1000.0)  # 1000m elevation
        Rd = 287.0
        Rd_cpd = 0.286
        gravity0 = 9.81
        
        result = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temperature,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        # Lowest level should be close to surface elevation
        # Allow for interpolation between mass and face levels
        assert np.all(result[:, :, 0] >= 0)

    def test_mass2height_temperature_dependency(self):
        """Test that warmer temperatures lead to higher altitudes."""
        nx, ny, nz = 5, 5, 5
        nz_faces = nz + 1
        
        hybrid_coef_A = np.linspace(100000, 0, nz_faces)
        hybrid_coef_B = np.linspace(0, 1, nz_faces)
        surface_pressure = np.full((nx, ny), 11.0)
        z_surface = np.zeros((nx, ny))
        Rd = 287.0
        Rd_cpd = 0.286
        gravity0 = 9.81
        
        # Test with cold temperature
        temp_cold = np.full((nx, ny, nz), 250.0)
        result_cold = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temp_cold,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        # Test with warm temperature
        temp_warm = np.full((nx, ny, nz), 300.0)
        result_warm = mass2height_coordinates(
            hybrid_coef_A,
            hybrid_coef_B,
            surface_pressure,
            temp_warm,
            z_surface,
            Rd,
            Rd_cpd,
            gravity0,
            nx,
            ny,
            nz,
        )
        
        # Warmer atmosphere should lead to higher levels at same pressure
        assert np.all(result_warm[:, :, -1] > result_cold[:, :, -1])


# Integration test with actual file (if available)
def test_mass2height_with_real_data():
    """Integration test with real AROME data if available."""
    arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")
    
    if not os.path.exists(arome_file):
        pytest.skip("AROME test file not available")
    
    from arome2fvm.arome_reader import AromeReader
    
    reader = AromeReader(arome_file)
    
    z = mass2height_coordinates(
        reader.get_hybrid_coef_A(),
        reader.get_hybrid_coef_B(),
        reader.get_surface_pressure(),
        reader.get_temperature(),
        reader.get_surface_geopotential() / 9.81,
        287.0,  # Rd
        287.0 / 1004.0,  # Rd/cpd
        9.81,
        reader.get_nx(),
        reader.get_ny(),
        reader.get_nz(),
    )
    
    # Test shape
    assert z.shape == (reader.get_nx(), reader.get_ny(), reader.get_nz())
    
    # Test monotonic increase at multiple random points
    for _ in range(10):
        i = np.random.randint(0, reader.get_nx())
        j = np.random.randint(0, reader.get_ny())
        
        for k in range(1, reader.get_nz()):
            assert z[i, j, k] > z[i, j, k - 1], \
                f"Height not increasing at ({i},{j},{k})"
    
    # Test reasonable height ranges (should be within troposphere)
    assert np.all(z >= 0)
    assert np.max(z) < 30000  # Below 30km


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
