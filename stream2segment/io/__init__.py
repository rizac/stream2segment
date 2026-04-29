from os.path import join, normpath, isabs, isdir, abspath, dirname
import yaml

# FIXME REMOVE!
def yaml_load(filepath, **updates):
    """Load a yaml file into a dict (if `filepath` is a `dict`, skips loading).
    Then:
    1. If `filepath` denotes a file path (and not a dict),
       normalizes non-absolute sqlite path values relative to `filepath`, if any
    2. updates the dict values with `updates` and returns the yaml dict. The
       update is recursive, meaning that nested dict values will be updated
       recursively and not completely overwritten. Example:
       param. a = {'b': 1, 'c' :2}, updates a = {'b': 2, 'd' :3} =>
       result = {'b': 2, 'c': 2, 'd': 3}

    :param filepath: str, dict or file-like object. If str, it must denote a
        path to an existing .yaml file
    :param updates: arguments which will updates the yaml dict before it is
        returned
    """
    if isinstance(filepath, str):
        with open(filepath, 'r') as stream:
            ret = yaml.safe_load(stream)
    elif hasattr(filepath, 'read'):
        ret = yaml.safe_load(filepath)
    else:
        raise TypeError(
            f'required file path (string) or file object, found: {type(filepath)}'
        )

    # update recursively (which means sub-dicts are updated as well and not
    # overwritten):
    def update(dic1, dic2):
        """update dic1 with dic2 recursively"""
        # Terminology: If the same key exists in both dicts and is mapped to
        # two dictionaries (not necessarily equal), the latter are called
        # "shared dicts"

        # 1. Move shared dicts from `dic2` in a temporary dictionary 'dickeys':
        dickeys = {
            k: dic2.pop(k) for k in dic1.keys()
            if isinstance(dic1[k], dict) and isinstance(dic2.get(k, None), dict)
        }
        # 2. `dic1` and `dic2` have no shared dicts, update dicts "normally":
        dic1.update(dic2)
        # 3. Update shared dicts recursively:
        for k in dickeys:
            update(dic1[k], dickeys[k])

    update(ret, updates)

    if isinstance(filepath, str):
        # convert relative sqlite path to absolute, assuming they are relative
        # to the config:
        sqlite_prefix = 'sqlite:///'
        # we cannot modify a dict while in iteration, thus create a new dict of
        # possibly modified sqlite paths and use later dict.update
        new_dict = {}
        for key, val in ret.items():
            try:
                if val.startswith(sqlite_prefix) and ":memory:" not in val:
                    db_path = val[len(sqlite_prefix):]
                    if not isabs(db_path):
                        db_path2 = abspath(join(dirname(filepath), db_path))
                        if db_path2 != db_path:
                            new_dict[key] = sqlite_prefix + db_path2
            except AttributeError:
                pass

        ret.update(new_dict)
    return ret
