import gc
import sys
from typing import Tuple

import torch
from torch import nn


def hook_fn(m, i, o):
    if i[0].shape[-1] == 412:
        print("*****START*****")
    if i[0].shape != o.shape or True:
        print(type(m), "\n", i[0].shape, "\n", o.shape)


def add_shape_hook(input: torch.Tensor, network: nn.Module):
    for _, layer in network._modules.items():
        if hasattr(layer, "children") and len(list(layer.children())) > 0:
            add_shape_hook(input, layer)
        else:
            layer.register_forward_hook(hook_fn)


class debug_cuda_context:
    """ Debug context to print cuda allocation between calls inside the context """

    def __init__(self, name):
        self.name = name
        self.objects = self.new_objects

    def __enter__(self):
        print("Entering Cuda Debug Decorated func")
        # Set the trace function to the trace_calls function
        # So all events are now traced
        try:
            sys.settrace(self.trace_calls)
        except TypeError as e:
            print(f"Cannot trace due to TypeError: {e}")

    def __exit__(self, *args, **kwargs):
        # Stop tracing all events
        sys.settrace = None

    def trace_calls(self, frame, event, arg):
        if event != "call":
            return
        elif frame.f_code.co_name != self.name:
            return
        return self.trace_lines

    @property
    def new_objects(self):
        if not hasattr(self, "objects"):
            return self.get_objects()
        with torch.no_grad():
            current_objects = self.get_objects()
            new_objects = current_objects.difference(self.objects)
            self.objects = current_objects
            return new_objects

    def get_objects(self):
        objects = set()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    objects.add(obj)
            except:
                pass
        return objects

    def trace_lines(self, frame, event, arg):
        if event not in ["line", "return"]:
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        GB = float(1024 ** 3)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        print(
            " {0} {1} {2} allocated: {3:.4f} GB".format(
                func_name, event, line_no, allocated / GB
            )
        )
        print("   New tensor allocations:")

        for obj in self.new_objects:
            print("   ", type(obj), obj.size())


def debug_cuda(func):
    """ Debug decorator to call the function within the debug context """

    def decorated_func(*args, **kwargs):
        context = debug_cuda_context(func.__name__)
        context.__enter__()
        try:
            return_value = func(*args, **kwargs)
        except RuntimeError as e:
            print("New tensors @ exception")
            for obj in context.new_objects:
                print(type(obj), obj.size())
            raise e
        context.__exit__()
        return return_value

    return decorated_func


def delete(*objs):
    for obj in objs:
        del obj
