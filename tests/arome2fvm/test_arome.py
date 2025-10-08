# -*- coding: utf-8 -*-
import os
from arome import Arome


def test_arome():
    
    arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")

    config_file = os.path.join(os.getcwd(), "config", "alps.yml")

    arome = Arome(arome_file, config_file)
    
