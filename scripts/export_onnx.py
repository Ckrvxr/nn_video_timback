import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.av1_vsr import AV1VSR


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prev, cur, next_):
        return self.model(prev, cur, next_, scale=4)


def main():
    config = yaml.safe_load(open('configs/test_full.yaml'))
    device = 'cuda'

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)

    ckpt = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    prev = torch.randn(1, 3, 540, 960, device=device)
    cur = torch.randn(1, 3, 540, 960, device=device)
    next_ = torch.randn(1, 3, 540, 960, device=device)

    torch.onnx.export(
        wrapper,
        (prev, cur, next_),
        'checkpoints/model.onnx',
        input_names=['prev', 'cur', 'next'],
        output_names=['output'],
        dynamic_axes={
            'prev': {2: 'height', 3: 'width'},
            'cur': {2: 'height', 3: 'width'},
            'next': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
        },
        opset_version=17,
    )
    print('Exported: checkpoints/model.onnx')


if __name__ == '__main__':
    main()
