# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from arome_reader import AromeReader

file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")
reader = AromeReader(file)


def test_get_nx():
    assert reader.get_nx() == 1013


def test_get_ny():
    assert reader.get_ny() == 757


def test_get_nz():
    assert reader.get_nz() == 90


def test_get_spacing():
    spacings = reader.get_spacing()
    assert spacings[0] == 1250
    assert spacings[1] == 1250


def get_surface_velocities():
    arr1, arr2 = reader.get_surface_velocities()
    assert arr1.shape == (1013, 757)
    assert arr2.shape == (1013, 757)


def get_surface_geopotential():
    arr = reader.get_surface_geopotential()
    assert arr.shape == (1013, 757)


def get_surface_pressure():
    arr = reader.get_surface_pressure()
    assert arr.shape == (1013, 757)


def get_vertical_divergence():
    arr = reader.get_vertical_divergence()
    assert arr.shape == (1013, 757, 91)


def get_temperature():
    arr = reader.get_temperature()
    assert arr.shape == (1013, 757, 91)


def get_pressure():
    arr = reader.get_pressure()
    assert arr.shape == (1013, 757, 91)


def get_hybrid_coef_A():
    arr = reader.get_hybrid_coef_A()
    assert arr.shape == 91


def get_hybrid_coef_B():
    arr = reader.get_hybrid_coef_B()
    assert arr.shape == 91
