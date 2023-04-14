from typing import Optional
from pathlib import Path

from x2r.io import FileSystemConfig


class Prepare:
    def __init__(
        self,
        output_dir: str,
        cache_dir: Optional[str] = None,
        filesystem: Optional[FileSystemConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.filesystem = filesystem

    def run(self) -> None:
        ...
