# -*- coding: utf-8 -*-
import logging
import logging.config
import time
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(Path.cwd(), "src")))
sys.path.append(str(Path(Path.cwd().parent.absolute(), "FVM_GT4Py_slim", "src")))
print(sys.path)

import fvms.model.config
import fvms.model.fields
from fvms.genutils.log import init_logging
from fvms.model.driver import model_driver
from arome2fvm.arome import Arome
from arome2fvm.writer import write_state

logging.getLogger(__name__)

def run():
    """Run model from an Arome initial state

    Returns:
        _type_: _description_
    """

    arome_file = str(Path(Path.cwd().parent.absolute(), "files", "historic.arome.nc"))
    config_file = str(Path(Path.cwd(), "config", "alps.yml"))
    data_file = str(Path(Path.cwd(), "config", "arome.nc"))
    
    arome = Arome(config_file=config_file, arome_file=arome_file)
    write_state(arome, data_file)

    fields = fvms.model.fields.FieldContainer(arome.fvm_config.indices)

    logging.info("Running model")

    time0 = time.time()
    model_driver(arome.fvm_config, fields)
    time1 = time.time()
    runtime = time1 - time0
    log.info(f"Execution time:  {runtime}")

    return None


if __name__ == "__main__":
    log = init_logging()
    log = logging.getLogger(__name__)
    run()
    log.info(f"Finished test")