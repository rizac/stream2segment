from stream2segment.resources import get_resource_abspath

def test_notebook(data):
    import os
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    # cwd = os.getcwd()
    fle = get_resource_abspath('templates', 'jupyter.example.ipynb')
    with open(fle) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)  # , kernel_name='python3')
    cwd = os.path.dirname(fle)
    ep.preprocess(nb, {'metadata': {'path': cwd}})