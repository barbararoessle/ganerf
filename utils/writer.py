import argparse
import copy
import json
from pathlib import Path

def write_output_json(exp_name, method_name, ckpt_path, metrics_dict, json_path):
    benchmark_info = {
        "experiment_name": exp_name,
        "method_name": method_name,
        "checkpoint": ckpt_path,
        "results": metrics_dict,
    }
    # Save output to output file
    Path(json_path).write_text(json.dumps(benchmark_info, indent=2), "utf8")

def namespace_to_dict(namespace):
    namespace_as_dict = copy.deepcopy(vars(namespace))  # make deep copy
    for k, v in namespace_as_dict.items():
        if isinstance(v, argparse.Namespace):
            namespace_as_dict[k] = namespace_to_dict(v)
    return namespace_as_dict


def dict_to_namespace(dictionary):
    d = copy.deepcopy(dictionary)  # make deep copy
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return argparse.Namespace(**d)

def print_network_info(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Number of model parameters: %.3f M" % (num_params / 1e6))