"""Script module generating wiki pages from current Jupyter notebook

WARNING: This script module not used. The generation fo the wiki pages is issued by
means of normal commands on the terminal (see the "Updating wiki" section in the README).
"""
from os.path import isdir

from stream2segment.resources import get_resource_abspath

for fle_ in ['jupyter.example.ipynb', 'the-segment-object.ipynb']:
    fle = get_resource_abspath('templates', fle_)
    with open(fle) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)  # , kernel_name='python3')
    cwd = os.path.dirname(fle)
    ep.preprocess(nb, {'metadata': {'path': cwd}})

if __name__ == "__main__":
    import sys
    argv = sys.argv
    if len(argv) < 2:
        print('Please provide the directory of the stream2segment wiki git repo')
    repo = argv[1]
    if not isdir(repo):
        print('"%s" is not an existing directory')

