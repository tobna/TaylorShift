import logging
import math
import traceback
from contextlib import nullcontext
from math import prod

import torch
from fvcore.nn import FlopCountAnalysis
from torchprofile import profile_macs
import torch.distributed as dist


def macs(args, model, input, n_ims=1):
    """Calculates the MACs (multiply-accumulate operations) of the model for a given input

    Parameters
    ----------
    args
        training arguments
    n_ims : int
        number of images to look at
    model : torch.nn.Module
        the model
    input : torch.Tensor
        the input tensor in batch format

    Returns
    -------
    int
        number of MAC operations needed
    """

    if n_ims is not None:
        input = input[:n_ims]
    with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
        return profile_macs(model, input)


def max_mem_allocated(device, world_size=1, reset_max=False):
    """Returns the max memory allocated during training.

    Use **this before** calling *throughput*, as that resets the statistics.

    Parameters
    ----------
    device : torch.Device
        the device to look at in this process
    world_size : int
        the number of GPUs (processes) used in total -> stats are gathered from all GPUs
    reset_max : bool
        if true, resets the max memory allocated to zero. Subsequent calls will return the max memory allocated after
        this call.

    Returns
    -------
    int | tuple[int, int]
        If world_size==1, returns the max memory allocated (in MB) on this GPU. Else, returns the max memory allocated
        (in B) on all GPUs together (sum) and the max memory allocated on any one GPU (max).
    """

    max_mem_gpu = torch.cuda.max_memory_allocated(device)
    if reset_max:
        torch.cuda.reset_peak_memory_stats(device)

    if world_size == 1:
        return max_mem_gpu

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, max_mem_gpu)
    return sum(gathered), max(gathered)


def inference_memory(args, model, input, device, batch_sizes=[1, 32, 64, 128]):
    """Returns the memory needed for inference at different batch sizes

    Parameters
    ----------
    args
        training arguments; in particular set args.eval_amp
    model : torch.nn.Module
        the model to evaluate
    input : torch.Tensor
        batch of input data; no batch size bigger than the size of this batch are tested
    device : torch.Device
        the device to test on
    batch_sizes : list[int]
        list of batch sizes to test

    Returns
    -------
    dict[int, int]
        the mapping batch size |-> vram needed

    """
    vram_allocated = {}

    for bs in sorted(batch_sizes, reverse=True):
        if input.shape[0] < bs:
            continue
        input = input[:bs]
        # reset statistics
        torch.cuda.reset_peak_memory_stats(device)

        with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
            with torch.no_grad():
                try:
                    model(input)
                    vram_allocated[bs] = max_mem_allocated(device, reset_max=True)
                except torch.cuda.OutOfMemoryError:
                    pass
    return vram_allocated


def _add_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs[0])
    # print(in_shapes, out_shape)
    # assert in_shapes[0][1:] == in_shapes[1][1:] and (in_shapes[0][0] == in_shapes[1][0] or in_shapes[1][0] == 1), \
    #     f"Got incompatible shapes for adding: {in_shapes}"
    return prod(out_shape)


def _mul_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)
    # assert len(in_shapes[1]) <= 1 or len(in_shapes[0]) <= 1 or in_shapes[1][1:] == [1, 1] or (len(in_shapes[0]) == len(in_shapes[1]) and all(x == y == out or (x == 1 and y == out) or (y == 1 and x == out) for x, y, out in zip(in_shapes[0], in_shapes[1], out_shapes[0]))), \
    #     f"mul_handler found in_shapes: {in_shapes} -> {out_shapes[0]}"
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")
    return prod(out_shapes[0])


def _softmax_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")

    # approximate times 5 for flops from exp, sum, and mult (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shapes[0]) * 5


def _gelu_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs[0])

    # approximate times * 8 for mult, add, tanh, and pow (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shape) * 8


def _div_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)

    return prod(out_shapes[0])


def _norm_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]

    # flops come from squaring each input (M*N) and adding all of them up (M*N - 1)
    norm_dims = [1]
    batch_dims = [1]
    for dim in set(in_shapes):
        if dim == 1:
            continue
        in_cnt = in_shapes.count(dim)
        out_cnt = out_shapes.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shapes}) then in shape ({in_shapes})"
        batch_dims += [dim for _ in range(out_cnt)]
        norm_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (2 * prod(norm_dims) - 1)


def _cumsum_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]
    assert out_shapes == in_shapes, f"cumsum: {out_shapes} != {in_shapes}"

    # assume worst case
    cumsum_dim = max(in_shapes)
    # in cumsum_dim: 0 + 1 + ... + n-1 = n(n-1)/2
    # for each of the batch dims (entries prod(all_dims) / cumsum_dim
    return int(prod(in_shapes) * (cumsum_dim - 1) / 2)


def _pow_handler(inputs, outputs):
    out_shapes = _get_cval_shape(outputs)[0]

    # print(f"pow map: {in_shapes} -> {out_shapes}")

    # for now assume pow <= 4 -> ~ 3 mults
    return 3 * prod(out_shapes)


def _sin_cos_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]

    # approximate each of these operations (on GPU) to be just 1 FLOP
    # taken from https://foldingathome.org/support/faq/flops/
    return prod(out_shape)


def _log_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _exp_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _sigmoid_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]

    # approximation: number of flops for exp + 2
    return 2 * prod(out_shape) + _exp_handler(inputs, outputs)


def _sum_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    sum_dims = [1]
    batch_dims = [1]
    for dim in set(in_shape):
        if dim == 1:
            continue
        in_cnt = in_shape.count(dim)
        out_cnt = out_shape.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shape}) then in shape ({in_shape})"
        batch_dims += [dim for _ in range(out_cnt)]
        sum_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (prod(sum_dims) - 1)


def _rfft2_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = in_shape[d_i_1] * in_shape[d_i_2]
    return int(prod(in_shape) * 2.5 * math.log2(N))


def _irfft2_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 2.5 * math.log2(N))


def _fft2_handler(inputs, outputs):
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 5 * math.log2(N))


def _mean_handler(inputs, outputs):
    in_shape = _get_cval_shape(inputs)[0]

    # mean of N elements takes N flops (N-1 for sum and 1 to divide by len)
    return prod(in_shape)


def _avg_pool2d_handler(inputs, outputs):
    # take the mean; the same way as in mean_handler
    return _mean_handler(inputs, outputs)


def _get_cval_shape(val):
    """Get the shapes from a jit value object.

    Taken from https://github.com/facebookresearch/fvcore/blob/fd5043ff8b2e6790f5bd7c9632695c68986cc658/fvcore/nn/jit_handles.py#L23

    Parameters
    ----------
    val : torch._C.Value | list[torch._C.Value]
        jit value object or list of those.

    Returns
    -------
    list[int] | list[list[int]]
        return a list of ints -> shape, or list of shapes.
    """
    if isinstance(val, list):
        return [_get_cval_shape(x) for x in val]

    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


def flops(args, model, input, per_module=False, n_ims=1):
    """Returns the number of floating point operations (FLOPs) needed for a given input.

    This function is broken, when working with timm models -> returns 0.
    The output should in theory be 2*MACs(), but it might report MACs straight up...
    Further investigation needed.

    Parameters
    ----------
    args
        training arguments; in particular set args.eval_amp
    n_ims : int
        number of images to look at
    model : torch.nn.Module
        the model to analyze
    input : torch.Tensor
        the input to give to the model
    per_module : bool
        flag to return stats by submodule

    Returns
    -------
    int | dict[str, int]
        total number of flops needed (in GFLOPs); either summed over all modules or by module (empty string -> total).
    """

    if n_ims is not None:
        input = input[:n_ims]

    fca = FlopCountAnalysis(model, input)
    fca.set_op_handle("aten::add", _add_handler)
    fca.set_op_handle("aten::add_", _add_handler)
    fca.set_op_handle("aten::mul", _mul_handler)
    fca.set_op_handle("aten::mul_", _mul_handler)
    fca.set_op_handle("aten::softmax", _softmax_handler)
    fca.set_op_handle("aten::gelu", _gelu_handler)
    fca.set_op_handle("aten::bernoulli_", None)
    fca.set_op_handle("aten::div_", _div_handler)
    fca.set_op_handle("aten::div", _div_handler)
    fca.set_op_handle("aten::norm", _norm_handler)
    fca.set_op_handle("aten::cumsum", _cumsum_handler)
    fca.set_op_handle("aten::pow", _pow_handler)
    fca.set_op_handle("aten::sin", _sin_cos_handler)
    fca.set_op_handle("aten::cos", _sin_cos_handler)
    fca.set_op_handle("aten::sum", _sum_handler)
    fca.set_op_handle("aten::fft_rfft2", _rfft2_handler)
    fca.set_op_handle("aten::fft_irfft2", _irfft2_handler)
    fca.set_op_handle("aten::fft_fft2", _fft2_handler)
    fca.set_op_handle("aten::mean", _mean_handler)
    fca.set_op_handle("aten::sub", _add_handler)
    fca.set_op_handle("aten::rsub", _add_handler)
    fca.set_op_handle("aten::reciprocal", _div_handler)
    fca.set_op_handle("aten::avg_pool2d", _avg_pool2d_handler)
    fca.set_op_handle("aten::adaptive_avg_pool1d", _avg_pool2d_handler)
    fca.set_op_handle("aten::log", _log_handler)
    fca.set_op_handle("aten::exp", _exp_handler)
    fca.set_op_handle("aten::sigmoid", _sigmoid_handler)
    fca.set_op_handle("aten::scatter_add", _add_handler)
    fca.set_op_handle("aten::log_softmax", _softmax_handler)
    fca.set_op_handle("aten::square", _mean_handler)

    # these operations are ignored, because 0 FLOPS
    fca.set_op_handle("aten::expand_as", None)
    fca.set_op_handle("aten::clamp_min", None)
    fca.set_op_handle("aten::view_as_complex", None)
    fca.set_op_handle("aten::real", None)
    fca.set_op_handle("aten::eye", None)
    fca.set_op_handle("aten::repeat_interleave", None)
    fca.set_op_handle("aten::scatter_reduce", None)
    fca.set_op_handle("aten::fill_", None)
    fca.set_op_handle("aten::ones_like", None)
    fca.set_op_handle("aten::topk", None)

    with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
        if per_module:
            return fca.by_module()
        try:
            return fca.total()
        except IndexError as e:
            logging.error(f"IndexError {e} when calculating flops. Might come from timm model.")
            traceback.print_exc()
            return -1


def throughput(args, model, input, device, iters=100):
    """Calculates the throughput of a given model.

    Throughput is given for the biggest batch_size, that fits into memory. Images from input are repeated to get to this
    batch_size.
    Internally resets the max allocated memory, so only use this **after** *max_mem_allocated*.

    Parameters
    ----------
    args
        training arguments; in particular set args.eval_amp
    model : torch.nn.Module
        the model to analyze
    input : torch.Tensor
        the batch of images to start with
    device : torch.cuda.device
        the device to measure throughput with
    iters : int
        the number of iterations to test with (for more accurate numbers)

    Returns
    -------
    tuple[int, float]
        model throughput at optimal *batch_size* in *images/second*.
    """

    bs = min(1024, args.batch_size)
    # print(f"test batch sizes: {test_bs}")
    if input.shape[0] < bs:
        input = torch.cat((input for _ in range(int(bs / input.shape[0]) + 1)), dim=0)
    input = input[:bs]

    results = []
    while True:
        logging.info(f"thoughput calculation: test batch size {bs}")
        try:
            tp = measure_throughput(model, input, iters, args.eval_amp)
        except RuntimeError as e:
            if "canUse32BitIndexMath" in str(e):
                logging.info(f"throughput calculation: tensor too large @ {bs}")
            else:
                logging.info(f"throughput calculation: CUDA OOM @ {bs}")
            break
        results.append((bs, tp))
        input = torch.cat((input, input), dim=0)
        bs *= 2
    return max(results, key=lambda x: x[1]) if len(results) > 0 else (-1, -1)


def measure_throughput(model, input, iters=1000, eval_amp=False):
    """
    Measures the throughput of a PyTorch model on CUDA.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to measure throughput for.
    input : torch.Tensor
        The input tensor of shape (batch_size, ...) for the model.
    iters : int, optional
        The number of iterations to run to measure the throughput, by default 1000.
    eval_amp : bool, optional
        Whether to evaluate using Automatic Mixed Precision (AMP) mode, by default False.

    Returns
    -------
    float
        The number of samples processed per second (throughput) on CUDA.

    """
    # total_time = 0
    samples = []
    with torch.no_grad():
        with torch.cuda.amp.autocast() if eval_amp else nullcontext():
            for _ in range(iters):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                __ = model(input)
                ender.record()
                torch.cuda.synchronize()
                # total_time += starter.elapsed_time(ender) / 1000  # ms -> s
                samples.append(starter.elapsed_time(ender) / 1000)  # ms -> s
    # return iters * input.shape[0]/total_time
    samples = samples[int(len(samples) / 10) :]
    return len(samples) * input.shape[0] / sum(samples)
