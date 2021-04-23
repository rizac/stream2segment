from os import listdir
from os.path import abspath, dirname, join, splitext

PATH = abspath(dirname(__file__))


def get_resource_abspath(*paths):
    """Return the resource file as the concatenation of this module path (resource
    directory) and any members of `*paths`.
    Same as `os.path.join(resource_directory, *paths)`

    :param paths:filename(s) relative to the resource directory
    """
    return join(PATH, *paths)


def get_ttable_fpath(basename):
    """Return the file for the given travel times table

    :param basename: the file name (with or without extension) located under
        `get_traveltimestables_dirpath()`
    """
    if not splitext(basename)[1]:
        basename += ".npz"
    return join(get_resource_abspath("traveltimes"), basename)


def get_templates_fpaths(*filenames):
    """Return the template file paths with given filename(s) inside the package
    `templates` of the `resource` directory. If filenames is empty (no
    arguments), returns all files (no dir) in the `templates` directory

    :param filenames: a list of file names relative to the templates directory.
        With no argument,returns all valid files inside that directory
    """
    templates_path = get_resource_abspath("templates")
    if not filenames:
        filenames = listdir(templates_path)

    return list(join(templates_path, _name) for _name in filenames)


def get_templates_fpath(filename):
    """Return the template file path with given filename inside the package
    `templates` of the `resource` directory

    :param filename: a filename relative to the templates directory
    """
    return get_templates_fpaths(filename)[0]