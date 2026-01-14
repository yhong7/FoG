
import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    # Python 内置随机
    random.seed(seed)
    # 环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU

    # 确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    def test_randomness():
        py_rand = [random.random() for _ in range(5)]
        np_rand = np.random.rand(5).tolist()
        torch_rand_cpu = torch.rand(5).tolist()
        if torch.cuda.is_available():
            torch_rand_cuda = torch.rand(5, device="cuda").tolist()
        else:
            torch_rand_cuda = None
        return {
            "python_random": py_rand,
            "numpy_random": np_rand,
            "torch_cpu_random": torch_rand_cpu,
            "torch_cuda_random": torch_rand_cuda,
        }

    set_all_seeds(42)
    r_1 = test_randomness()
    set_all_seeds(42)
    r_2 = test_randomness()
    assert r_1 == r_2


