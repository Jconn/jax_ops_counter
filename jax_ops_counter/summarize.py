import jax
import flax.linen as nn
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
import numpy as np
import flax.linen.module as module_lib
from flax.core import unfreeze
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from dataclasses import dataclass
import logging

def _get_path_variables(
    path: Tuple[str, ...], variables):
  """A function that takes a path and a variables structure and returns the variable structure at
  that path."""
  path_variables = {}

  for collection in variables:
    collection_variables = variables[collection]
    for name in path:
      if name not in collection_variables:
        collection_variables = None
        break
      collection_variables = collection_variables[name]

    if collection_variables is not None:
      path_variables[collection] = unfreeze(collection_variables)

  return path_variables

def get_module_variables(
    path: Tuple[str, ...],
    variables,
    all_paths):
  """A function that takes a path and variables structure and returns a

  (module_variables, submodule_variables) tuple for that path.
  _get_module_variables
  uses the `all_paths` set to determine if a variable belongs to a submodule or
  not.
  """
  module_variables = _get_path_variables(path, variables)
  submodule_variables: Any = {collection: {} for collection in module_variables}
  all_keys = set(
      key for collection in module_variables.values() for key in collection
  )

  for key in all_keys:
    submodule_path = path + (key,)
    if submodule_path in all_paths:
      for collection in module_variables:
        if key in module_variables[collection]:
          submodule_variables[collection][key] = module_variables[
              collection
          ].pop(key)

  return module_variables, submodule_variables

def gru_handler(call_data, variables, all_paths):
    def unpack_call(call_data):
        inputs = call_data.args
        outputs = call_data.outputs
        assert len(inputs) == 2, "gru layers need two inputs"
        hidden_shape = inputs[0].shape
        ishape = inputs[1].shape
        dtype = inputs[0].dtype
        oshape = outputs[0].shape
        output_hidden_shape = outputs[1].shape 
        this_module, child_modules = get_module_variables(call_data.path, variables, all_paths)
        params_dict = child_modules['params']
        all_params = 0
        for k,inner_dict in params_dict.items():
            all_params += sum_params(inner_dict)


        return all_params, ishape, oshape, hidden_shape, dtype
    all_params, ishape, oshape, hidden_shape, dtype = unpack_call(call_data)
    assert len(ishape) == 3, "GRU layers only support 3d inputs"
    batch = ishape[0]
    timesteps = ishape[1]

    macs = all_params * timesteps
    #matmul
    macs_per_output = ishape[-1]
    #again for fully connected total loads is simple
    total_loads = 2 * macs
    return macs, all_params, macs_per_output, total_loads 



def sum_params(params_dict): 
    all_params = 0
    for k,v in params_dict.items():
        all_params += np.prod(v.shape)
    return all_params

def conv_handler(call_data,variables, all_paths):
    def unpack_call(call_data):
        inputs = call_data.args
        assert len(inputs) == 1, "dense layers only one arg only supports one input"
        conv_input = inputs[0]
        conv_output = call_data.outputs

        ishape = conv_input.shape
        dtype = conv_input.dtype
        oshape = conv_output.shape
        this_module, child_modules = get_module_variables(call_data.path, variables, all_paths)
        params_dict = this_module['params']
        all_params = sum_params(params_dict)
        kernel = params_dict['kernel'].shape
        return all_params, ishape, oshape, dtype, kernel


    all_params, ishape, oshape, dtype, kernel_shape = unpack_call(call_data) 
    macs = np.prod(kernel_shape) * np.prod(oshape[1:3])
    macs_per_output = np.prod(kernel_shape)
    total_loads = all_params + macs 

    return macs, all_params, macs_per_output, total_loads 


def fc_handler(call_data,variables, all_paths):
    def unpack_call(call_data):
        inputs = call_data.args
        assert len(inputs) == 1, "dense layers only one arg only supports one input"
        fc_input = inputs[0]
        fc_output = call_data.outputs
        ishape = fc_input.shape
        dtype = fc_input.dtype
        oshape = fc_output.shape
        this_module, child_modules = get_module_variables(call_data.path, variables, all_paths)
        params_dict = this_module['params']
        all_params = sum_params(params_dict)
        return all_params, ishape, oshape, dtype
    all_params, ishape, oshape, dtype = unpack_call(call_data)
    #input channels, output channels, output shape, kernel size
    macs = ishape[-1] * np.prod(oshape[1:])
    macs_per_output = ishape[-1] 
    total_loads = all_params +  macs
    return macs, all_params, macs_per_output, total_loads 

    batch_size = ishape[0]
    assert batch_size == oshape[0], "misinterpretation of shapes {} {}".format(ishape, oshape)
    num_outputs = np.prod(oshape[1:])
    #assumption: fully connected active dimensions is just the last one
    #num macs is going to be the 
    params = oshape[-1] * ishape[-1]
    #params is also the inner macs
    outer_repititions = 1
    if len(oshape) > 2:
        outer_repititions = np.prod(oshape[1:-1])
    macs = outer_repititions * params
    macs_per_output = ishape[-1] 
    total_loads = all_params + macs
    return macs, all_params, macs_per_output, total_loads 

def module_handler(module_type):
    module_handler = {}
    module_handler[nn.Dense] = fc_handler
    module_handler[nn.Conv] = conv_handler
    if module_type in module_handler.keys():
        return module_handler[module_type]
    return None

@dataclass 
class LayerData:
    path: str
    module_type: str
    inputs: List
    outputs: List
    macs: int
    params: int
    macs_per_output: int
    total_loads: int

    def get_summable_values(self):
        return self.macs, self.params, self.macs_per_output, self.total_loads

    def set_summable_values(self, macs, params, macs_per_output, total_loads):
        self.macs = macs
        self.params = params
        self.macs_per_output = macs_per_output
        self.total_loads = total_loads


def get_model_stats(model, *model_args):
    def get_variables():
        return model.init(*model_args) # Initialization call
    def flatten(x):
        output = []
        if isinstance(x, tuple) or isinstance(x,list):
            for inst in x:
                output.extend(flatten(inst))
        else:
            output = [x]
        return output



    with module_lib._tabulate_context():
        variables = jax.eval_shape(get_variables)
        calls = module_lib._context.call_info_stack[-1].calls
        calls.sort(key=lambda c: c.index)
        
    #calls contains the ordered calls for the network
    all_paths: Set[Tuple[str, ...]] = set(call.path for call in calls)
    all_data = []
    def safe_shape(x):
        try:
            return x.shape
        except:
            return ()
    for call in calls:
        handler = module_handler(call.module_type)
        if isinstance(call.outputs, tuple):
            outputs = [safe_shape(x) for x in flatten(call.outputs)]
        else:
            outputs = [safe_shape(call.outputs)]
        if handler is None:
            #check to make sure that the module is not a leaf
            macs, all_params, macs_per_output, total_loads = [0] * 4
        else:
            macs, all_params, macs_per_output, total_loads = handler(call, variables,all_paths)
        def sanitized_shape(x):
            try:
                return x.shape
            except:
                return None
        row = LayerData(call.path, call.module_type, [sanitized_shape(x) for x in flatten(call.args)],\
                outputs, macs, all_params, macs_per_output, total_loads)


                
        all_data.append(row)

    def sum_values(base_layer, next_layers):
        
        #base layer is also a tuple
        #grab the latest thing in baselayer
        def is_leaf():
            if next_layers is None:
                return True
            if base_layer == next_layers[-1]:
                return True

        is_root = False
        if len(base_layer.path) > 0:
            base_name = base_layer.path[-1]
        else:
            base_name = None
            is_root = True

        child_layers = []
        for idx, layer in enumerate(next_layers):
            #the root layer is the first layer and represents the entire model
            def is_child(x):
                x_path = x.path
                #root children are one layer deep
                if is_root:
                    return len(x_path) == 1
                #reject different parents
                #reject grandchildren
                if len(x_path) >= 2 and x_path[:-1] == base_layer.path:
                    return True
                return False 

            if is_child(layer):
                sum_values(layer, next_layers[idx+1:]) 
                #problem - I'm considering grandchildren as children
                #children are not contiguous
                child_layers.append(layer)
                
        if len(child_layers) == 0:
            #no children assume values have already been computed 
            return
        summable_values = base_layer.get_summable_values()
        for layer in child_layers:
            #sum the values
            summable_values = [x + y for x,y in zip(summable_values, layer.get_summable_values())]
        logging.debug("summing values for {}".format(base_layer.path))
        base_layer.set_summable_values(*summable_values)

    sum_values(all_data[0], all_data[1:])
    return all_data

def pretty_table(data_array, depth=None):
    from prettytable import PrettyTable
    # Get field names dynamically from data class
    field_names = [field.name for field in data_array[0].__dataclass_fields__.values()]
    if depth is not None:
        depth_array =[x for x in data_array if len(x.path) <= depth]
        data_array = depth_array

    table = PrettyTable()
    table.field_names = field_names

    for obj in data_array:
        table.add_row([getattr(obj, field) for field in field_names])

    #TODO -deal with hardcoding
    table.align["path"] = "l"
    table.align["module_type"] = "l"
    table.align["inputs"] = "l"
    table.align["outputs"] = "l"
    table.align["total_loads"] = "r"
    table.align["macs"] = "r"
    table.align["params"] = "r"
    table.align["macs_per_output"] = "r"
    return table


