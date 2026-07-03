from contextlib import contextmanager
from os.path import dirname
from pathlib import Path

import pytest
import sys
import os
from stream2segment.process.main import redirect


@contextmanager
def legacy_redirect(src=None, dst=os.devnull):
    """Prevent Python AND external C shared library to print to stdout/stderr in Python,
    preventing also leaking file descriptors.
    If the first argument is None or any object not having a fileno() argument, this
    context manager is simply no-op and will yield and then return

    See (in this order):
    https://stackoverflow.com/a/14797594
    and (final solution modified here):

    Example:

    with redirect(sys.stdout):
        print("from Python")
        os.system("echo non-Python applications are also supported")

    :param src: file-like object with a fileno() method. Usually is either `sys.stdout`
        or `sys.stderr`.
    """
    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything
    # Moreover, passing None as first argument means no redirection
    if src is None:
        yield
        return

    try:
        file_desc = src.fileno()
    except (AttributeError, OSError, ValueError) as _:
        yield
        return

    # if you want to assert that Python and C stdio write using the same file descriptor:
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == file_desc == 1

    def _redirect_stderr_to(fileobject):
        sys.stderr.close()  # + implicit flush()
        # make `file_desc` point to the same file as `fileobject`.
        # First closes file_desc if necessary:
        os.dup2(fileobject.fileno(), file_desc)
        # Make Python write to file_desc
        sys.stderr = os.fdopen(file_desc, 'w')

    def _redirect_stdout_to(fileobject):
        sys.stdout.close()  # + implicit flush()
        # make `file_desc` point to the same file as `fileobject`.
        # First closes file_desc if necessary:
        os.dup2(fileobject.fileno(), file_desc)
        # Make Python write to file_desc
        sys.stdout = os.fdopen(file_desc, 'w')

    _redirect_to = _redirect_stderr_to if src is sys.stderr else _redirect_stdout_to

    with os.fdopen(os.dup(file_desc), 'w') as src_fileobject:
        with open(dst, 'w') as dst_fileobject:
            _redirect_to(dst_fileobject)
        try:
            yield  # allow code to be run with the redirected stdout/err
        finally:
            # restore stdout/err. buffering and flags such as CLOEXEC may be different:
            _redirect_to(src_fileobject)




# @pytest.fixture(autouse=True)
# def disable_capture(capfd):
#     with capfd.disabled():
#         yield

def _c_write(fd: int, message: bytes):
    """Simulate a C shared library writing directly to a raw fd,
    bypassing Python's buffering layer entirely."""
    os.write(fd, message)


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_c_writes_go_to_dst(stream_name, tmp_path):
    """C-level writes to the redirected fd land in dst, not on the terminal."""
    src = sys.stdout if stream_name == "stdout" else sys.stderr
    fd = src.fileno()
    dst = tmp_path / "dst.txt"
    dst.touch()

    with redirect(src, dst=dst):
        _c_write(fd, b"c-library noise\n")

    with open(dst, 'rb') as f:
        assert b"c-library noise" in f.read()


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_fd_restored_after_block(stream_name, tmp_path):
    """After the block, the fd is restored and writes work normally again."""
    src = sys.stdout if stream_name == "stdout" else sys.stderr
    fd = src.fileno()
    dst_during = tmp_path / "during.txt"
    dst_after = tmp_path / "after.txt"
    dst_during.touch()
    dst_after.touch()

    with redirect(src, dst=dst_during):
        _c_write(fd, b"inside\n")

    # fd is now restored: redirect again to a new file to verify
    with redirect(src, dst=dst_after):
        _c_write(fd, b"after restore\n")

    with open(dst_after, 'rb') as f:
        assert b"after restore" in f.read()


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_python_wrapper_is_same_object_after_restore(stream_name, tmp_path):
    """The original Python stream object is restored (same object, no new wrapper)."""
    src = sys.stdout if stream_name == "stdout" else sys.stderr
    obj_before = src
    dst = tmp_path / "dst.txt"
    dst.touch()

    with redirect(src, dst=dst):
        pass

    obj_after = sys.stdout if stream_name == "stdout" else sys.stderr
    assert obj_after is obj_before


def test_noop_when_src_is_none():
    """redirect(None) is a no-op."""
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    with redirect(None):
        pass
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before


def test_noop_when_no_fileno(capsys):
    """redirect with a src that has no fileno() (e.g. under pytest) is a no-op."""
    # under pytest sys.stderr has no real fd; redirect should not raise
    with redirect(sys.stderr, dst=os.devnull):
        print("visible", file=sys.stdout)
    assert "visible" in capsys.readouterr().out


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_fd_restored_after_exception(stream_name, tmp_path):
    """fd is restored even when an exception is raised inside the block."""
    src = sys.stdout if stream_name == "stdout" else sys.stderr
    fd = src.fileno()

    try:
        with redirect(src, dst=os.devnull):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    dst = tmp_path / "after_exc.txt"
    dst.touch()

    with redirect(src, dst=dst):
        _c_write(fd, b"restored\n")

    with open(dst, 'rb') as f:
        assert b"restored" in f.read()


if __name__ == "__main__":
    import time
    import subprocess

    print("You should see 'ls: /does/not/exist: No such file or directory':")
    subprocess.Popen(["ls", "/does/not/exist"])
    time.sleep(3)
    print("You should **NOT** see 'ls: /does/not/exist: No such file or directory':")
    with redirect(sys.stderr):
        subprocess.Popen(["ls", "/does/not/exist"])
        #time.sleep(2)
    print('Done, exiting (no line above me right?)')