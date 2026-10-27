"""Bounded test worker for MLIR's C ABI; process exit reclaims bufferized temporaries."""

import ctypes
import json
import sys
from pathlib import Path

import numpy as np

library, schema, inputs_path, outputs_path = map(Path, sys.argv[1:])
signature = json.loads(schema.read_text())


def descriptor(rank):
    class MemRef(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRef


with np.load(inputs_path) as archive:
    arrays = [np.array(archive[name], copy=True, order="C") for name in signature["inputs"]]
arguments = []
for array, shape in zip(arrays, signature["input_shapes"], strict=True):
    rank = len(shape)
    kind = descriptor(rank)
    arguments.append(
        kind(
            array.ctypes.data,
            array.ctypes.data,
            0,
            (ctypes.c_int64 * rank)(*shape),
            (ctypes.c_int64 * rank)(*(s // 4 for s in array.strides)),
        )
    )
outputs = [descriptor(len(shape)) for shape in signature["output_shapes"]]
if len(outputs) == 1:
    container = outputs[0]()
    descriptors = [container]
else:

    class Results(ctypes.Structure):
        _fields_ = [(f"out{i}", kind) for i, kind in enumerate(outputs)]

    container = Results()
    descriptors = [getattr(container, f"out{i}") for i in range(len(outputs))]
shared = ctypes.CDLL(str(library))
function = shared._mlir_ciface_main
function.restype = None
function.argtypes = [ctypes.POINTER(type(container)), *[ctypes.POINTER(type(a)) for a in arguments]]
function(ctypes.byref(container), *[ctypes.byref(arg) for arg in arguments])
result = {}
for name, shape, value in zip(
    signature["outputs"], signature["output_shapes"], descriptors, strict=True
):
    assert list(value.sizes) == shape
    assert value.offset >= 0 and all(s >= 0 for s in value.strides)
    span = value.offset + sum((d - 1) * s for d, s in zip(shape, value.strides, strict=True)) + 1
    assert span <= 16 * 1024 * 1024
    pointer = ctypes.cast(value.aligned, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(pointer, shape=(span,))
    result[name] = np.lib.stride_tricks.as_strided(
        data[value.offset :], shape=tuple(shape), strides=tuple(4 * s for s in value.strides)
    ).copy()
np.savez(outputs_path, **result)
