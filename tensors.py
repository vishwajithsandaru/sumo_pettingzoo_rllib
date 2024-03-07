from tensordict import TensorDict
import torch


# tensor_dict = TensorDict({
#     'tensor1': torch.Tensor([1, 3]),
#     'tensor2': torch.Tensor([[[1, 3, 3], [1, 3, 4], [1, 3, 4]], [[1, 3, 3], [1, 3, 4], [1, 3, 4]]]),
#     'tensor3': torch.Tensor([1, 3]),
# },batch_size=torch.Size([]))

# print(tensor_dict)

logits = torch.Tensor([])