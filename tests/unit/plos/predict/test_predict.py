import numpy as np
import torch
from torch import device
from torch_geometric.data import Batch, Data

from eclair.plos.model.model import Binary_model, MinkUNet34
from eclair.plos.predict.predict import predict_model


def test_predict_model_supports_binary_model() -> None:
    model = Binary_model(MinkUNet34(in_channels=3, out_channels=1))
    point_count = 2
    data = Data(
        classification=torch.tensor([0, 0], dtype=torch.uint8),
        rgb=torch.tensor([[0, 0, 0], [0, 0, 0]]),
        index_quantile_excluded=torch.tensor([], dtype=torch.int64),
        index_kept=torch.tensor([0, 1]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]], dtype=torch.float64),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        y=torch.tensor([0, 0], dtype=torch.uint8),
        batch=torch.tensor([0, 0]),
        ptr=torch.tensor([0, 2]),
    )
    batch = Batch.from_data_list([data])
    labels = predict_model(model, batch, voxel_size=0.05, device=device("cpu"))
    assert labels.shape == (point_count,)
    assert labels.dtype == np.uint8
