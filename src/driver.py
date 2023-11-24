import typer

from arome2fvm import AromeConfig

def main(arome_file: str,
        config_file: str,
        data_file: str
        ):
        
    # Reads AROME file
    config = AromeConfig(arome_file)
    
    if config_file is not None:
        config.to_file(config_file)

    if data_file is not None:
        config.write_data(data_file)

    
if __name__ ==  "__main__":
    typer.run(main)