from dataclasses import dataclass
from functools import cached_property

@dataclass(frozen=True)
class PhysicalConstants:
    
    # Vertical temperature gradient
    Rd: float = 287.059674
    p0: float = 1000.0e2
    gravity0: float = 9.80665
    cpd: float = 1004.709
    
    @cached_property
    def Rd_cpd(self):
        return self.Rd / self.cpd

PhysicalConstants()

