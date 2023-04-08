from typing import Optional

from pyarrow.fs import FileSystem, LocalFileSystem

_global_filesystem = LocalFileSystem()


def set_global_filesystem(filesystem: Optional[FileSystem]):
    global _global_filesystem

    if filesystem is None:
        filesystem = LocalFileSystem()

    _global_filesystem = filesystem


def get_global_filesystem() -> FileSystem:
    return _global_filesystem
