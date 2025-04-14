import os


def create_dir(DirOut):
    """
    Create a directory if it does not exist.

    Args:
        DirOut (str): The directory path to create.
    """

    # Check if the directory exists, if not, create it
    isExist = os.path.exists(DirOut)
    if not isExist:
        os.makedirs(DirOut)
    return DirOut