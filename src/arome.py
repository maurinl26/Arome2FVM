""" Initialization from AROME initial conditions
    (netCDF4 file converted from a FA file)
    Including orography and vertical coordinate over 90 levels
"""
import netCDF4 as nc
from functools import cached_property
from typing import TYPE_CHECKING

from fvms.build_config import dtype
from fvms.utils.storage import to_numpy
import numpy as np

if TYPE_CHECKING:
    from fvms.model.config import Config
    from fvms.utils.storage import Field
    from fvms.utils.typingx import Triple



class AROME:
    
    def __init__(self, config: Config):
        
        # Constants
        self.p0 = config.constants.p0
        self.Rd_cpd = config.constants.Rd_cpd
        
        # Coordinates
        self.xcr = to_numpy(config.coordinates.xcr["covering"])
        self.ycr = to_numpy(config.coordinates.ycr["covering"])
        self.zcr = to_numpy(config.coordinates.zcr["covering"])
        
        with nc.Dataset(config.file) as ds:
            
            nx = ds.dimensions['X']
            ny = ds.dimensions['Y']
            nz = ds.dimensions['Z']
            
            self.temperature = np.zeros((nx, ny, nz))
            self.pressure = np.zeros((nx, ny, nz))
            self.uvel = np.zeros((nx, ny, nz))
            self.vvel = np.zeros((nx, ny, nz))
            self.generalized_vertical_velocity = np.zeros((nx, ny, nz))
            
            # Temperature 
            # TEMPERATURE
            self.temperature = np.zeros((nx, ny, nz))
            for i in range(nz):
                self.temperature[:, :, i] = ds[f"S{i:0>3}TEMPERATURE"][...]
            
            # Pressure 
            # PRESS.DEPART
            for i in range(nz):
                self.pressure[:, :, i] = ds[f"S{i:0>3}PRESS.DEPART"][...]
            
            # U 
            # WIND.U.PHYS
            for i in range(nz):
                self.uvel[:, :, i] = ds[f"S{i:0>3}WIND.U.PHYS"][...]            
            
            # V 
            # WIND.V.PHYS
            for i in range(nz):
                self.vvel[:, :, i] = ds[f"S{i:0>3}WIND.V.PHYS"][...]
            
            # Vertical divergence for generalized velocity
            # VERTIC.DIVER
            for i in range(nz):
                self.generalized_vertical_velocity[:, :, i] = ds[f"S{i:0>3}VERTIC.DIVER"][...]

            
            
    def set_zcr(self, zcr: np.ndarray) -> np.ndarray:
        
        # Convert sigma hybrid coordinate to z following terrain coordinate
        
        return None
    
    def set_theta(self, theta: Field) -> None:
        theta = self.temperature * (self.p0 /self.pressure) ** (self.Rd_cpd)
    
    def set_exner(self, exner: Field) -> None:
        exner = (self.pressure / self.p0) ** self.Rd_cpd
    
    def set_rho(self, rho: Field) -> None:
        return None
    
    def set_rvapour(self, rv: Field) -> None:
        return None
    
    def set_velocity(self, uvel: Triple[Field]) -> None:
        
        # Convert divergence for vertical velocity to vertical velocity
    
        return None
