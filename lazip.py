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
# along with lazip.  If not, see <https://www.gnu.org/licenses/>.

"""Lazy ZIP over HTTP"""

__version__ = '0.0.4'
__all__ = ['Filazy', 'Lazip']

from abc import abstractmethod
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from io import UnsupportedOperation
from tempfile import NamedTemporaryFile
from typing import IO, Dict, Iterator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

from requests import Session
from requests.models import CONTENT_CHUNK_SIZE, Response


class ReadOnlyBinaryIOWrapper(IO[bytes]):
    """Wrapper for a read-only binary I/O."""

    file: IO[bytes]
    length: int

    @property
    def mode(self) -> str:
        """Opening mode, which is always rb."""
        return 'rb'

    @property
    def name(self) -> str:
        """File name."""
        return self.file.name

    def close(self) -> None:
        """Close the file."""
        self.file.close()

    @property
    def closed(self) -> bool:
        """Whether the file is closed."""
        return self.file.closed

    def fileno(self) -> int:
        """Return the underlying file descriptor (an integer)."""
        return self.file.fileno()

    def flush(self) -> None:
        """Do nothing."""
        self.file.flush()

    def isatty(self) -> bool:
        """Return False."""
        return self.file.isatty()

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the object and return them.

        As a convenience, if size is unspecified or -1,
        all bytes until EOF are returned.  Fewer than
        size bytes may be returned if EOF is reached.
        """
        start = self.tell()
        stop = start + size if 0 <= size <= self.length-start else self.length
        self.ensure(start, stop-1)
        return self.file.read(size)

    def readable(self) -> bool:
        """Return True."""
        return self.file.readable()

    def readline(self, limit):
        raise UnsupportedOperation

    def readlines(self, hint):
        raise UnsupportedOperation

    def seek(self, offset: int, whence: int = 0) -> int:
        """Change stream position and return the new absolute position.

        Seek to offset relative position indicated by whence:
        * 0: Start of stream (the default).  pos should be >= 0;
        * 1: Current position - pos may be negative;
        * 2: End of stream - pos usually negative.
        """
        return self.file.seek(offset, whence)

    def seekable(self) -> bool:
        """Return whether random access is supported, which is True."""
        return self.file.seekable()

    def tell(self) -> int:
        """Return the current possition."""
        return self.file.tell()

    def truncate(self, size: Optional[int] = None) -> int:
        """Resize the stream to the given size in bytes.

        If size is unspecified resize to the current position.
        The current stream position isn't changed.

        Return the new file size.
        """
        return self.file.truncate(size)

    def writable(self) -> bool:
        """Return False."""
        return False

    def write(self, s):
        raise UnsupportedOperation

    def writelines(self, lines):
        raise UnsupportedOperation

    def __next__(self):
        raise UnsupportedOperation

    def __iter__(self):
        raise UnsupportedOperation

    def __enter__(self) -> 'ReadOnlyBinaryIOWrapper':
        self.file.__enter__()
        return self

    def __exit__(self, *exc) -> Optional[bool]:
        return self.file.__exit__(*exc)

    @abstractmethod
    def ensure(self, start: int, end: int) -> None:
        """Ensure the data from start to end inclusively.

        This method must return to the original position
        if seek is called.
        """


class Filazy(ReadOnlyBinaryIOWrapper):
    """Read-only file-like object mapped to a file over HTTP.

    This uses HTTP range requests to lazily fetch the file's content.
    At the end of initialization, __post_init__ will be called.

    Parameters:
        session (Session): Requests session
        url (str): HTTP URL to the file
        chunk_size (int): Download chunk size

    Attributes:
        session (Session): Requests session
        url (str): HTTP URL to the file
        chunk_size (int): Download chunk size
        left (List[int]): Left endpoints of downloaded intervals
        right (List[int]): Right endpoints of downloaded intervals
        accept_ranges (bool): Whether range requests are supported
    """

    def __init__(self, session: Session, url: str,
                 chunk_size: int = CONTENT_CHUNK_SIZE) -> None:
        response = session.head(url)
        response.raise_for_status()
        assert response.status_code == 200
        headers = response.headers
        self.session, self.url, self.chunk_size = session, url, chunk_size
        self.length = int(headers['Content-Length'])
        self.file = NamedTemporaryFile()
        self.truncate(self.length)
        self.left: List[int] = []
        self.right: List[int] = []
        self.accept_ranges = 'bytes' in headers.get('Accept-Ranges', 'none')
        with self.stay(): self.__post_init__()

    def __post_init__(self) -> None:
        pass

    @contextmanager
    def stay(self) -> Iterator[None]:
        """Return a context manager that keeps the stream position.

        At the end of the block, seek back to original position.
        """
        pos = self.tell()
        try:
            yield
        finally:
            self.seek(pos)

    def stream_response(self, start: int, end: int,
                        base_headers: Dict[str, str] = {}) -> Response:
        """Return HTTP response to a range request from start to end."""
        headers = {'Range': f'bytes={start}-{end}', **base_headers}
        return self.session.get(self.url, headers=headers, stream=True)

    def merge(self, start: int, end: int,
              left: int, right: int) -> Iterator[Tuple[int, int]]:
        """Return an iterator of intervals to be fetched.

        Parameters:
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

    def ensure(self, start: int, end: int) -> None:
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


class Lazip(Filazy):
    """Read-only file-like object mapped to a ZIP file over HTTP.

    This uses HTTP range requests to lazily fetch the file's content,
    which is supposed to be fed to ZipFile.
    """

    def __post_init__(self) -> None:
        """Check and download until the file is a valid ZIP."""
        end = self.length - 1
        if not self.accept_ranges:
            self.ensure(0, end)
            self.left, self.right = [0], [end]
            return
        for start in reversed(range(0, end, self.chunk_size)):
            self.ensure(start, end)
            try:
                ZipFile(self)
            except BadZipFile:
                pass
            else:
                break
