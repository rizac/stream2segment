'''
Created on Apr 23, 2016

@author: riccardo
'''
import os
from os import strerror, errno
import shutil


def ensure(filepath, mode, mkdirs=False, error_type=OSError):
    """checks for filepath according to mode, raises an Exception instanceof error_type if the check
    returns false
    :param mode: either 'd', 'dir', 'r', 'fr', 'w', 'fw' (case insensitive). Checks if file_name is,
        respectively:
            - 'd' or 'dir': an existing directory
            - 'fr', 'r': file for reading (an existing file)
            - 'fw', 'w': file for writing (basically, an existing file or a path whose dirname
            exists)
    :param mkdirs: boolean indicating, when mode is 'file_w' or 'dir', whether to attempt to
        create the necessary path. Ignored when mode is 'r'
    :param error_type: The error type to be raised in case (defaults to OSError. Some libraries
        such as ArgumentPArser might require their own error
    :type error_type: any class extending BaseException (OsError, TypeError, ValueError etcetera)
    :raises: SyntaxError if some argument is invalid, or error_type if filepath is not valid
        according to mode and mkdirs
    :return: True if mkdir has been called
    """
    # to see OsError error numbers, see here
    # https://docs.python.org/2/library/errno.html#module-errno
    # Here we use two:
    # errno.EINVAL ' invalid argument'
    # errno.errno.ENOENT 'no such file or directory'
    if not filepath:
        raise error_type("{0}: '{1}' ({2})".format(strerror(errno.EINVAL),
                                                   str(filepath),
                                                   str(type(filepath))
                                                   )
                         )

    keys = ('fw', 'w', 'fr', 'r', 'd', 'dir')

    # normalize the mode argument:
    if mode.lower() in keys[2:4]:
        mode = 'r'
    elif mode.lower() in keys[:2]:
        mode = 'w'
    elif mode.lower() in keys[4:]:
        mode = 'd'
    else:
        raise error_type('{0}: mode argument must be in {1}'.format(strerror(errno.EINVAL),
                                                                    str(keys)))

    if errmsgfunc is None:  # build custom errormsgfunc if None
        def errmsgfunc(filepath, mode):
            if mode == 'w' or (mode == 'r' and not os.path.isdir(os.path.dirname(filepath))):
                return "{0}: '{1}' ({2}: '{3}')".format(strerror(errno.ENOENT),
                                                        os.path.basename(filepath),
                                                        strerror(errno.ENOTDIR),
                                                        os.path.dirname(filepath)
                                                        )
            elif mode == 'd':
                return "{0}: '{1}'".format(strerror(errno.ENOTDIR), filepath)
            elif mode == 'r':
                return "{0}: '{1}'".format(strerror(errno.ENOENT), filepath)

    if mode == 'w':
        to_check = os.path.dirname(filepath)
        func = os.path.isdir
        mkdir_ = mkdirs
    elif mode == 'd':
        to_check = filepath
        func = os.path.isdir
        mkdir_ = mkdirs
    else:  # mode == 'r':
        to_check = filepath
        func = os.path.isfile
        mkdir_ = False

    exists_ = func(to_check)
    mkdirdone = False
    if not func(to_check):
        if mkdir_:
            mkdirdone = True
            os.makedirs(to_check)
            exists_ = func(to_check)

    if not exists_:
        raise error_type(errmsgfunc(filepath, mode))

    return mkdirdone


# def ensurefiler(filepath):
#     """Checks that filepath denotes a valid file, raises an OSError if not. This function is mostly
#     useful for initializing filepaths given as input argument (e.g. command line) also in conjunction
#     with libraries such as e.g. click, OptionParser or ArgumentParser.
#     For instance, it raises a meaningful OSError in case of non-existing parent directory (hopefully
#     saving useless browsing time). For any other case, it might be more convenient to simply call the
#     almost equivalent os.path.isfile(filepath)
#     :param filepath: a file path
#     :type filepath: string
#     :return: nothing
#     :raises: OSError if filepath does not denote an existing file
#     """
#     _ensure(filepath, 'r', False)  # last arg ignored, set to False for safety
# 
# 
# def ensurefilew(filepath, mkdirs=True):
#     """Checks that filepath denotes a valid file for writing, i.e., if its parent directory D
#     exists. Raises an OSError if not. This function is mostly useful for initializing filepaths given as
#     input argument (e.g. command line) also in conjunction with libraries such as e.g. click,
#     OptionParser or ArgumentParser.
#     :param filepath: a file path
#     :type filepath: string
#     :param mkdirs: True by default, if D does not exists will try to build it via mkdirs before
#         re-checking again its existence
#     :return: nothing
#     :raises: OSError if filepath directory does not denote an existing directory
#     """
#     _ensure(filepath, 'w', mkdirs)
# 
# 
# def ensuredir(filepath, mkdirs=True):
#     """Checks that filepath denotes a valid existing directory. Raises an OSError if not. This
#     function is mostly useful for initializing filepaths given as input argument (e.g. command line)
#     also in conjunction with libraries such as e.g. click, OptionParser or ArgumentParser.
#     For any other case, it might be more convenient to
#     call the almost equivalent os.path.isdir(filepath)
#     :param filepath: a file path
#     :type filepath: string
#     :param mkdirs: True by default, if D does not exists will try to build it via mkdirs before
#         re-checking again its existence
#     :return: nothing
#     :raises: OSError if filepath directory does not denote an existing directory
#     """
#     _ensure(filepath, 'd', mkdirs)


def rsync(source, dest, update=True, modify_window=1):
    """
    Copies source to dest emulating a simple rsync unix command
    :param source: the source file. If it does not exist, an OSError is raised
    :param dest: the destination file. According to shutil.copy2, if dest is a directory then
    the destination file will be os.path.join(dest, os.basename(source)
    :param update: If True (the default), the copy will be skipped for a file which exists on
        the destination and has a modified time that is newer than the source file.
        (If an existing destination file has a modification time equal to the source file's,
        it will be updated if the sizes are different.)
    :param modify_window: (1 by default). This argument is ignored if update is False. Otherwise,
        when comparing two timestamps, this function treats the timestamps as being equal if they
        differ by no more than the modify-window value. This is normally 0 (for an exact match),
        but it defaults to 1 as (quoting from rsync docs):
        "In particular, when transferring to or from an MS Windows FAT filesystem
         (which represents times with a 2-second resolution), --modify-window=1 is useful
         (allowing times to differ by up to 1 second).
        If update is a float, or any object parsable to float (e.g. "4.5"), it will be rounded to
        integer
    :return: the tuple (destination_file, copied), where the first item is the destination file
    (which might not be the dest argument, if the latter is a directory) and a boolean denoting if
    the copy has been performed. Note that it is not guaranteed that the returned file exists (the
    user has to check for it)
    """
    if not os.path.isfile(source):
        raise OSError(strerror(errno.ENOENT) + ": '" + source + "'")

    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(source))

    if update and os.path.isfile(dest):
        st1, st2 = os.stat(source), os.stat(dest)
        # st# = (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
        mtime_src, mtime_dest = st1[8], st2[8]
        # ignore if
        # 1) dest is newer than source OR
        # 2) times are equal (i.e. within the specified interval) AND sizes are equal (sizes are
        # the stats elements at index 6)
        if mtime_dest > mtime_src + update or \
                (mtime_src - update <= mtime_dest <= mtime_src + update and st1[6] == st2[6]):
            return dest, False

    shutil.copy2(source, dest)
    return dest, True


