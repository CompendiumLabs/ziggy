# gguf helper

import numpy as np

class Guffy:
    def __init__(self, path):
        from gguf import GGUFReader
        self.gf = GGUFReader(path)
        self.fields = list(self.gf.fields.keys())
        self.tlocs = {t.name: i for i, t in enumerate(self.gf.tensors)}

    def __repr__(self):
        from gguf.constants import GGUFValueType
        width = max([len(f) for f in self.fields])
        lines = []
        for field in self.fields:
            ftype = self.get_field_type(field)
            if ftype is not GGUFValueType.ARRAY:
                value = self.get_field(field)
                lines.append(f'{field:{width}} = {value}')
            else:
                length = self.get_arr_len(field)
                artype = self.get_arr_type(field)
                lines.append(f'{field:{width}} = {length} x <{artype.name}>')
        return '\n'.join(lines)

    ##
    ## getting fields
    ##

    def get_fields(self):
        return self.fields

    def get_field_data(self, key):
        return self.gf.get_field(key)

    def get_field_type(self, name):
        return self.gf.get_field(name).types[0]

    def get_field(self, key):
        from gguf.constants import GGUFValueType
        num_types = [
            GGUFValueType.UINT8, GGUFValueType.INT8,
            GGUFValueType.UINT16, GGUFValueType.INT16,
            GGUFValueType.UINT32, GGUFValueType.INT32,
            GGUFValueType.UINT64, GGUFValueType.INT64,
            GGUFValueType.FLOAT32, GGUFValueType.FLOAT64,
            GGUFValueType.BOOL
        ]

        ftypes = self.get_field_data(key).types
        ftype = ftypes[0]

        if len(ftypes) == 1:
            if ftype == GGUFValueType.STRING:
                return self.get_field_str(key)
            elif ftype in num_types:
                return self.get_field_num(key)
        elif len(ftypes) == 2:
            if ftype == GGUFValueType.ARRAY:
                if ftypes[1] == GGUFValueType.STRING:
                    return self.get_arr_str(key)
                elif ftypes[1] in num_types:
                    return self.get_arr_num(key)

        raise ValueError(f'Unknown field type: {ftype}')\

    def get_field_str(self, key):
        entry = self.gf.get_field(key)
        loc, = entry.data
        bval = entry.parts[loc].tobytes()
        return bval.decode('utf-8')

    def get_field_num(self, key):
        entry = self.gf.get_field(key)
        loc, = entry.data
        val, = entry.parts[loc]
        return val

    def get_arr_len(self, key):
        entry = self.gf.get_field(key)
        return len(entry.data)

    def get_arr_type(self, key):
        entry = self.gf.get_field(key)
        return entry.types[1]

    def get_arr_str(self, key):
        entry = self.gf.get_field(key)
        return [
            entry.parts[loc].tobytes().decode('utf-8') for loc in entry.data
        ]

    def get_arr_num(self, key):
        entry = self.gf.get_field(key)
        return np.array([
            entry.parts[loc][0] for loc in entry.data
        ])

    ##
    ## getting tensors
    ##

    def get_tensors(self):
        return list(self.tlocs)

    def get_tensor_meta(self, name):
        return self.gf.tensors[self.tlocs[name]]

    def get_tensor_type(self, name):
        return self.get_tensor_meta(name).type

    def get_tensor_shape(self, name):
        return self.get_tensor_meta(name).shape

    def get_tensor(self, name):
        info = self.get_tensor_meta(name)
        shape = list(reversed(info.shape))
        return np.asarray(info.data).reshape(shape)
