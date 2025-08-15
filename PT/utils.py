import gc
import torch

def list_tensors_on_gpu(model=None, min_numel=1_000_000):
    print("\n📦 [GPU Tensor Report]")
   
    # Build a map of id(param) -> name
    param_name_map = {}
    if model is not None:
        for name, param in model.named_parameters():
            param_name_map[id(param)] = name
            # Optionally track .data as well
            param_name_map[id(param.data)] = name + ".data"

    total_mem = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj if torch.is_tensor(obj) else obj.data
                if tensor.device.type == 'cuda':
                    numel = tensor.numel()
                    size_MB = tensor.element_size() * numel / 1e6
                    total_mem += size_MB

                    if numel >= min_numel:
                        name = param_name_map.get(id(tensor), "<unnamed>")
                        print(f"  • {type(obj).__name__:<20} "
                              f"shape={tuple(tensor.shape):<25} "
                              f"size={size_MB:.2f} MB "
                              f"requires_grad={tensor.requires_grad:<5} "
                              f"is_leaf={tensor.is_leaf:<5} "
                              f"name={name}")
        except Exception:
            pass

    print(f"\n🔢 Estimated Total GPU Tensor Memory: {total_mem:.2f} MB\n")

