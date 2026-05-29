from contextlib import contextmanager
from os.path import dirname
from pathlib import Path

import pytest
import sys
import os
# from stream2segment.process.main import redirect


@contextmanager
def redirect(src=None, dst: str | Path = os.devnull):
    """Redirect the OS-level file descriptor of `src` (sys.stdout or sys.stderr)
    to `dst` for the duration of the block. This silences C shared libraries that
    write directly to the underlying fd, while Python's own sys.stdout/sys.stderr
    objects are preserved and restored unchanged.

    No-op when:
      - src is None
      - src has no real fileno() (e.g. pytest's StringIO replacement for sys.stderr)

    :param src: sys.stdout or sys.stderr
    :param dst: destination path, default os.devnull
    """
    if src is None:
        yield
        return

    try:
        file_desc = src.fileno()
    except (AttributeError, OSError, ValueError):
        # pytest and similar tools replace sys.stderr/stdout with objects that
        # have no real file descriptor; treat as no-op
        yield
        return

    # Save the current Python wrapper so we can restore it exactly (same object,
    # no new wrapper created, no GC leak).
    # ORIGINAL BUG: created new wrappers via os.fdopen() on both redirect and restore
    old_stream = sys.stderr if src is sys.stderr else sys.stdout

    # Flush before touching the fd so no buffered Python output goes to dst.
    # Use the current stream object (old_stream), not src, which may be stale
    # if sys.stderr was already replaced (though here they are the same).
    old_stream.flush()

    # Save a duplicate of the original fd so we can restore it later.
    saved_fd = os.dup(file_desc)
    try:
        # Open dst and dup2 it onto file_desc, then close the temporary dst_fd.
        # ORIGINAL BUG: used `with open(dst) as dst_fileobject` which closed dst
        # before yield, leaving file_desc pointing to a closed fd during the block.
        dst_fd = os.open(dst, os.O_WRONLY)
        os.dup2(dst_fd, file_desc)
        os.close(dst_fd)
        # At this point file_desc points to dst at the OS level.
        # The existing Python wrapper (old_stream) still holds the same fd number
        # and will now write to dst — no new wrapper needed.

        try:
            yield
        finally:
            # Flush whatever is currently assigned to the stream (which is
            # old_stream, now writing to dst) before restoring.
            # ORIGINAL BUG: called src.flush() where src was the stale reference
            # captured at function entry, not the currently assigned stream.
            if src is sys.stderr:
                sys.stderr.flush()
            else:
                sys.stdout.flush()

            # Restore the original fd at OS level.
            os.dup2(saved_fd, file_desc)

            # Restore the original Python wrapper (same object, no new allocation).
            if src is sys.stderr:
                sys.stderr = old_stream
            else:
                sys.stdout = old_stream
    finally:
        os.close(saved_fd)


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
