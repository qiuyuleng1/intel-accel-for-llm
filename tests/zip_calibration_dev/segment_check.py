import sys

import torch

sys.path.insert(0, sys.argv[1])
from zip_calibration import generate_mock_kv_cache  # noqa: E402

from iaxl import torch_ext  # noqa: E402

TENSOR_COUNT = 32


def make_tensors(tensor_kb: int) -> list:
    data = generate_mock_kv_cache((2, 64, 512, 8, 128), "bf16").cpu().flatten()
    elements = tensor_kb * 1024 // data.element_size()
    tensors = []
    for index in range(TENSOR_COUNT):
        tensors.append(data[index * elements:(index + 1) * elements].clone())
    return tensors


def decompress_matches(blocks: list, expected: list) -> bool:
    outputs = []
    for tensor in expected:
        outputs.append(torch.empty_like(tensor))
    torch_ext.decompress_tensors(blocks, outputs)
    for index in range(len(expected)):
        if not torch.equal(expected[index], outputs[index]):
            return False
    return True


def main() -> None:
    mode = sys.argv[2]
    if mode == "roundtrip":
        expected = make_tensors(int(sys.argv[3]))
        inputs = []
        for tensor in expected:
            inputs.append(tensor.clone())
        blocks = torch_ext.compress_tensors(inputs)
        print("ROUNDTRIP", "PASS" if decompress_matches(blocks, expected) else "FAIL", flush=True)
    elif mode == "save":
        expected = make_tensors(int(sys.argv[3]))
        inputs = []
        for tensor in expected:
            inputs.append(tensor.clone())
        blocks = torch_ext.compress_tensors(inputs)
        torch.save({"expected": expected, "blocks": blocks}, sys.argv[4])
        print("SAVED", len(blocks), flush=True)
    elif mode == "load":
        saved = torch.load(sys.argv[4])
        result = "PASS" if decompress_matches(saved["blocks"], saved["expected"]) else "FAIL"
        print("LOAD_DECOMPRESS", result, flush=True)
    elif mode == "mock_ratio":
        data = generate_mock_kv_cache((2, 256, 512, 8, 128), "bf16")
        device = data.device.type
        data = data.cpu().flatten()
        elements = 256 * 1024 // data.element_size()
        inputs = []
        for index in range(data.numel() // elements):
            inputs.append(data[index * elements:(index + 1) * elements])
        blocks = torch_ext.compress_tensors(inputs)
        compressed_bytes = 0
        for block in blocks:
            compressed_bytes += block.numel()
        ratio = data.numel() * data.element_size() / compressed_bytes
        print(f"MOCK device {device} ratio {ratio:.4f}", flush=True)


if __name__ == "__main__":
    main()
