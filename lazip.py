# Lazy ZIP over HTTP
# Copyright (C) 2020  Nguyễn Gia Phong
#
# This file is part of lazip.
#
# lazip is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# lazip is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with palace.  If not, see <https://www.gnu.org/licenses/>.

"""Lazy ZIP over HTTP"""

__version__ = '0.0.2'
__all__ = ['Lazip']

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

from requests import Session
from requests.models import CONTENT_CHUNK_SIZE, Response


def init_range(stop: int, size: int) -> Iterator[Tuple[int, int]]:
    """Return an iterator of intervals to fetch a file reversedly."""
    start = stop - size
    while start > 0:
        yield start, stop-1
        stop = start
        start -= size
    yield 0, stop-1


class Lazip:
    """File-like object mapped to a ZIP file over HTTP.

    This uses HTTP range requests to lazily fetch the file's content,
    which is supposed to be fed to ZipFile.
    """

    def __init__(self, session: Session, url: str,
                 chunk_size: int = CONTENT_CHUNK_SIZE) -> None:
        head = session.head(url)
        head.raise_for_status()
        assert head.status_code == 200
        self.session, self.url, self.chunk_size = session, url, chunk_size
        self.length = int(head.headers['Content-Length'])
        self.file = NamedTemporaryFile()
        self.file.truncate(self.length)
        self.left: List[int] = []
        self.right: List[int] = []
        self.check_zip('bytes' in head.headers.get('Accept-Ranges', 'none'))

    def __enter__(self) -> 'Lazip':
        self.file.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Optional[bool]:
        return self.file.__exit__(*exc)

    @property
    def name(self) -> str:
        """File name."""
        return self.file.name

    def seekable(self) -> bool:
        """Return whether random access is supported, which is True."""
        return True

    @contextmanager
    def stay(self) -> Iterator[None]:
        """Return a context manager keeping the position.

        At the end of the block, seek back to original position.
        """
        pos = self.tell()
        try:
            yield
        finally:
            self.seek(pos)

    def check_zip(self, range_request: bool) -> None:
        """Check and download until the file is a valid ZIP."""
        if not range_request:
            end = self.length - 1
            self.download(0, end)
            self.left, self.right = [0], [end]
            return
        for start, end in init_range(self.length, self.chunk_size):
            self.download(start, end)
            with self.stay():
                try:
                    ZipFile(self)   # type: ignore
                except BadZipFile:
                    pass
                else:
                    break

    def stream_response(self, start: int, end: int,
                        base_headers: Dict[str, str] = {}) -> Response:
        """Return HTTP response to a range request from start to end."""
        headers = {'Range': f'bytes={start}-{end}', **base_headers}
        return self.session.get(self.url, headers=headers, stream=True)

    def merge(self, start: int, end: int,
              left: int, right: int) -> Iterator[Tuple[int, int]]:
        """Return an iterator of intervals to be fetched.

        Args:
            start (int): Start of needed interval
            end (int): End of needed interval
            left (int): Index of first overlapping downloaded data
            right (int): Index after last overlapping downloaded data
        """
        lslice, rslice = self.left[left:right], self.right[left:right]
        i = start = min([start, *lslice[:1]])
        end = max([end, *rslice[-1:]])
        for j, k in zip(lslice, rslice):
            if j > i: yield i, j-1
            i = k + 1
        if i <= end: yield i, end
        self.left[left:right], self.right[left:right] = [start], [end]

    def download(self, start: int, end: int) -> None:
        """Download bytes from start to end inclusively."""
        with self.stay():
            i, j = bisect_left(self.right, start), bisect_right(self.left, end)
            for start, end in self.merge(start, end, i, j):
                response = self.stream_response(start, end)
                response.raise_for_status()
                self.seek(start)
                for chunk in response.raw.stream(self.chunk_size,
                                                 decode_content=False):
                    self.file.write(chunk)

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the object and return them.

        As a convenience, if size is unspecified or -1,
        all bytes until EOF are returned.  Fewer than
        size bytes may be returned if EOF is reached.
        """
        start = self.tell()
        stop = start + size if 0 <= size <= self.length-start else self.length
        self.download(start, stop-1)
        return self.file.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Change stream position and return the new absolute position.

        Seek to offset relative position indicated by whence:
        * 0: Start of stream (the default).  pos should be >= 0;
        * 1: Current position - pos may be negative;
        * 2: End of stream - pos usually negative.
        """
        return self.file.seek(offset, whence)

    def tell(self) -> int:
        """Return the current possition."""
        return self.file.tell()

    def close(self) -> None:
        """Close the file."""
        self.file.close()
