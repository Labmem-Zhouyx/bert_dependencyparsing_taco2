import torch
import dgl


def get_mask_from_lengths(lengths, total_length):
    # use total_length of a batch for data parallelism
    device = lengths.device
    ids = torch.arange(0, total_length, out=torch.cuda.LongTensor(total_length)).to(device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(device)
    return mask


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def graph_to_device(g, device, graph_type):
    edge_tensor1 = g.edges()[0].to(device).long()
    edge_tensor2 = g.edges()[1].to(device).long()
    nodes_num = len(g.nodes())

    if graph_type == "fwd_type":
        # forward typed edges, top-down
        new_g = dgl.graph((edge_tensor2, edge_tensor1), num_nodes=nodes_num)
        new_g.edata['type'] = g.edata['type'].to(device).long()

    elif graph_type == "fwd_untype":
        # forward untyped edges, top-down
        new_g = dgl.graph((edge_tensor2, edge_tensor1), num_nodes=nodes_num)
        new_g.edata['type'] = torch.zeros_like(g.edata['type']).to(device).long()

    elif graph_type == "rev_type":
        # reverse typed edges, bottom-up
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = g.edata['type'].to(device).long()

    elif graph_type == "rev_untype":
        # reverse untyped edges, bottom-up
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = torch.zeros_like(g.edata['type']).to(device).long()

    elif graph_type == "bi_type":
        # bi-directional typed edges
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = g.edata['type'].to(device).long()

    elif graph_type == "bi_untype":
        # bi-directional untyped edges
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = torch.zeros_like(g.edata['type']).to(device).long()

    else:
        raise ValueError("Unsupported Dependency Graph type: {} ".format(graph_type))

    return new_g


_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *input, output_device=None):
    """Pytorch parallel processing

    Code from:
        https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer
    """
    global _output_ref
    global _replicas_ref

    device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    replicas = torch.nn.parallel.replicate(model, device_ids)

    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)

    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)

    y_hat = torch.nn.parallel.gather(outputs, output_device)

    _output_ref = outputs
    _replicas_ref = replicas

    return y_hat
