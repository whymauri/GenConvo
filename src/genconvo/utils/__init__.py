import logging

def get_logger(name: str, **kwargs) -> logging.Logger:
    import sys
    import os

    # Only add handler if we're on rank 0 or LOCAL_RANK is not set
    local_rank = os.environ.get("LOCAL_RANK", "0")
    
    logger = logging.getLogger(f"{name} [rank={local_rank}]", **kwargs)
    logger.setLevel(logging.INFO)

    if local_rank == "0" or True:
        handler = logging.StreamHandler(sys.stdout)  # Send logs to stdout
        handler.setLevel(logging.INFO)  # Set the log level for this handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Customize format
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger