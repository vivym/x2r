from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(kw_only=True)
class FileSystemConfig:
    _target_: str = "pyarrow.fs.FileSystem"


@dataclass(kw_only=True)
class LocalFileSystemConfig(FileSystemConfig):
    _target_: str = "pyarrow.fs.LocalFileSystem"


@dataclass(kw_only=True)
class S3FileSystemConfig(FileSystemConfig):
    _target_: str = "pyarrow.fs.S3FileSystem"

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    anonymous: bool = False
    role_arn: Optional[str] = None
    session_name: Optional[str] = None
    external_id: Optional[str] = None
    load_frequency: int = 900
    region: Optional[str] = None
    request_timeout: Optional[float] = None
    connect_timeout: Optional[float] = None
    scheme: str = "https"
    endpoint_override: Optional[str] = None
    background_writes: bool = True
    proxy_options: Optional[Union[str, Dict]] = None
    allow_bucket_creation: bool = False
    allow_bucket_deletion: bool = False
