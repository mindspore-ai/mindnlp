# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Download functions
"""

import os
import shutil
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse
import requests
from tqdm import tqdm


def get_cache_path():
    r"""
    Get the storage path of the default cache. If the environment 'cache_path' is set, use the environment variable.

    Args:
        None

    Returns:
        - **cache_dir**(str) - The path of default or the environment 'cache_path'.

    Examples:
        >>> default_cache_path = get_cache_path()
        >>> print(default_cache_path)
        '{home}\.text'

    """
    if "CACHE_DIR" in os.environ:
        cache_dir = os.environ.get("CACHE_DIR")
        if os.path.isdir(cache_dir):
            return cache_dir
        raise NotADirectoryError(f"{os.environ['CACHE_DIR']} is not a directory.")
    cache_dir = os.path.expanduser(os.path.join("~", ".text"))

    return cache_dir


def http_get(url, path=None, md5sum=None):
    r"""
    Download from given url, save to path.

    Args:
        url (str): download url
        path (str): download to given path (default value: '{home}\.text')

    Returns:
        - **cache_dir**(str) - The path of default or the environment 'cache_path'.

    Raises:
        TypeError: If `url` is not a String.
        RuntimeError: If `url` is None.

    Examples:
        >>> url = 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz'
        >>> cache_path = http_get(url)
        >>> print(cache_path)
        ('{home}\.text', '{home}\aclImdb_v1.tar.gz')

    """
    if path is None:
        path = get_cache_path()

    if not os.path.exists(path):
        os.makedirs(path)

    retry_cnt = 0
    retry_limit = 3
    name = os.path.split(url)[-1]
    filename = os.path.join(path, name)

    while not (os.path.exists(filename) and check_md5(filename, md5sum)):
        if retry_cnt < retry_limit:
            retry_cnt += 1
        else:
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        req = requests.get(url, stream=True, verify=False)
        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        tmp_filename = filename + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_filename, "wb") as f:
            if total_size:
                with tqdm(total=int(total_size), unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_filename, filename)

    return Path(path), filename


def check_md5(filename: str, md5sum=None):
    r"""
    Check md5 of download file.

    Args:
        filename (str) : The fullname of download file.
        md5sum (str) : The true md5sum of download file.

    Returns:
        - ** md5_check_result ** (bool) - The md5 check result.

    Raises:
        TypeError: If `filename` is not a string.
        RuntimeError: If `filename` is None.

    Examples:
        >>> filename = 'test'
        >>> check_md5_result = check_md5(filename)
        True

    """
    if md5sum is None:
        return True

    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    md5hex = md5.hexdigest()

    if md5hex != md5sum:
        return False
    return True


def get_dataset_url(datasetname: str):
    r"""
    Get dataset url for download

    Args:
        datasetname (str) : The name of the dataset to download.

    Returns:
        - ** url ** (str) - The url of the dataset to download.

    Raises:
        TypeError: If `datasetname` is not a string.
        RuntimeError: If `datasetname` is None.

    Examples:
        >>> name = 'aclImdb_v1'
        >>> url = get_dataset_url(name)
        >>> print(url)
        'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz'

    """
    default_dataset_json = "./mindtext/configs/dataset_url.json"
    with open(default_dataset_json, "r") as json_file:
        json_dict = json.load(json_file)

    url = json_dict.get(datasetname, None)
    if url:
        return url
    raise KeyError(f"There is no {datasetname}.")


def get_filepath(path: str):
    r"""
    Get the filepath of file.

    Args:
        path (str) : The path of the required file.

    Returns:
        - ** get_filepath_result ** (str) - If `path` is a folder containing a file, return `{path}\{filename}`;
                                        if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `path` is not a string.
        RuntimeError: If `path` is None.

    Examples:
        >>> path = '{home}\.text'
        >>> get_filepath_result = get_filepath(path)
        >>> print(get_filepath_result)
        '{home}\.text'

    """
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) == 1:
            return os.path.join(path, files[0])
        return path
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"{path} is not a valid file or directory.")


def cache_file(filename: str, cache_dir: str = None, url: str = None, md5sum=None):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        filename (str) : The name of the required dataset file.
        cache_dir (str) : The path of save the file.
        url (str) : The url of the required dataset file.

    Returns:
        - ** dataset_dir ** (str) - If `path` is a folder containing a file, return `{path}\{filename}`;
                                        if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `filename` is not a string.
        TypeError: If `cache_dir` is not a string.
        TypeError: If `url` is not a string.
        RuntimeError: If `filename` is None.

    Examples:
        >>> filename = 'aclImdb_v1'
        >>> path, filename = cache_file(filename)
        >>> print(path, filename)
        '{home}\.text' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        cache_dir = get_cache_path()
    if url is None:
        url = get_dataset_url(filename)
    path, filename = cached_path(
        filename_or_url=url, cache_dir=cache_dir, foldername=None, md5sum=md5sum
    )

    return path, filename


def cached_path(filename_or_url: str, cache_dir: str = None, foldername=None, md5sum=None):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        filename_or_url (str) : The name or url of the required file .
        cache_dir (str) : The path of save the file.
        name (str) : The name of the required dataset file.

    Returns:
        - ** Path ** (str) - If `path` is a folder containing a file, return `{path}\{filename}`;
                            if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `path` is not a string.
        RuntimeError: If `path` is None.

    Examples:
        >>> path = "https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz"
        >>> path, filename = cached_path(path)
        >>> print(path, filename)
        '{home}\.text\aclImdb_v1.tar.gz' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        dataset_cache = Path(get_cache_path())
    else:
        dataset_cache = cache_dir

    if foldername:
        dataset_cache = os.path.join(dataset_cache, foldername)

    parsed = urlparse(filename_or_url)

    if parsed.scheme in ("http", "https"):
        return get_from_cache(filename_or_url, Path(dataset_cache), md5sum=md5sum)
    if (parsed.scheme == "" and Path(os.path.join(dataset_cache, filename_or_url)).exists()):
        return Path(os.path.join(dataset_cache, filename_or_url))
    if parsed.scheme == "":
        raise FileNotFoundError("file {} not found in {}.".format(filename_or_url, dataset_cache))
    raise ValueError("unable to parse {} as a URL or as a local path".format(filename_or_url))


def match_file(filename: str, cache_dir: str) -> str:
    r"""
    If there is the file in cache_dir, return the path; otherwise, return empty string or error.

    Args:
        filename (str) : The name of the required file.
        cache_dir (str) : The path of save the file.

    Returns:
        - ** match_file_result ** (str) - If there is the file in cache_dir, return filename;
                                        if there is no such file, return empty string '';
                                        if there are two or more matching file, report an error.

    Raises:
        TypeError: If `filename` is not a string.
        TypeError: If `cache_dir` is not a string.
        RuntimeError: If `filename` is None.
        RuntimeError: If `cache_dir` is None.

    Examples:
        >>> name = 'aclImdb_v1.tar.gz'
        >>> path = get_cache_path()
        >>> match_file_result = match_file(name, path)
        ''

    """
    files = os.listdir(cache_dir)
    matched_filenames = []
    for file_name in files:
        if re.match(filename + "$", file_name) or re.match(filename + "\\..*", file_name):
            matched_filenames.append(file_name)
    if not matched_filenames:
        return ""
    if len(matched_filenames) == 1:
        return matched_filenames[-1]
    raise RuntimeError(f"Duplicate matched files:{matched_filenames}, this should be caused by a bug.")


def get_from_cache(url: str, cache_dir: str = None, md5sum=None):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        url (str) : The path to download the file.
        cache_dir (str) : The path of save the file.

    Returns:
        - ** path ** (str) - The path of save the downloaded file.
        - ** filename ** (str) - The name of downloaded file.

    Raises:
        TypeError: If `url` is not a string.
        TypeError: If `cache_dir` is not a Path.
        RuntimeError: If `url` is None.

    Examples:
        >>> path = "https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz"
        >>> path, filename = cached_path(path)
        >>> print(path, filename)
        '{home}\.text' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        cache_dir = Path(get_cache_path())
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)

    match_dir_name = match_file(filename, cache_dir)
    dir_name = filename
    if match_dir_name:
        dir_name = match_dir_name
    cache_path = cache_dir / dir_name
    if cache_path.exists() and check_md5(cache_path, md5sum):
        return get_filepath(cache_path), filename
    path = http_get(url, cache_dir, md5sum)[1]
    return Path(path), filename
