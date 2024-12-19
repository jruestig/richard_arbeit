def save_config_copy_easy(path_to_file: str, path_to_save_file: str):
    from shutil import copy, SameFileError
    try:
        copy(path_to_file, path_to_save_file)
        print(f"Config file saved to: {path_to_save_file}.")
    except SameFileError:
        pass
