# general utilities

from typing import Any
import toml
import operator

# use __reduce__ to handle pickle-dict malarky
class IndexedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = {k: i for i, k in enumerate(self.keys())}

    def __getstate__(self):
        return (dict(self), self._index)

    def __setstate__(self, state):
        data, self._index = state
        self.update(data)

    def __reduce__(self):
        return (IndexedDict, (), self.__getstate__())

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key not in self._index:
            self._index[key] = len(self._index)

    def __delitem__(self, name):
        super().__delitem__(name)
        del self._index[name]

    def update(self, *args, **kwargs):
        for d in (*args, kwargs):
            for k, v in d.items():
                self[k] = v

    def index(self, name):
        return self._index[name]

class Bundle(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for d in args + (kwargs,):
            self.update(d)

    @classmethod
    def from_tree(cls, tree):
        if isinstance(tree, dict):
            return cls([(k, cls.from_tree(v)) for k, v in tree.items()])
        else:
            return tree

    @classmethod
    def from_toml(cls, path):
        return cls.from_tree(toml.load(path))

    def __repr__(self):
        return '\n'.join([f'{k} = {v}' for k, v in self.items()])

    def keys(self):
        return sorted(super().keys())

    def items(self):
        return sorted(super().items(), key=operator.itemgetter(0))

    def values(self):
        return [k for k, _ in self.items()]

    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value
