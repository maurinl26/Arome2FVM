

from arome_file import AromeReader
from fvms.model.config import Config

class AromeConfig(Config):
    
    filename: str
    arome_reader: AromeReader
    
    def __init__(self):
        self.__init__(super)
        self.arome_reader = AromeReader(self.filename)
        

    def set_indices(self):
        NotImplemented
        
    def set_grid(self):
        self.grid
        
    def set_zcr(self):
        self.zcr
        
    def set_velocity(self):
        NotImplemented
        
 