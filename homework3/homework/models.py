from pathlib import Path

import torch
from torch._prims_common import Tensor
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class RegressionLoss(nn.Module):
  def forward(self, predictions: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss() 
    return loss(predictions,target)

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits,target)


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            if n_input != n_output:
                self.skip=(torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, padding=0)) 
            else:
                self.skip = torch.nn.Identity()
               
        def forward(self, x):
            # identity = x
            # if self.downsample is not None:
            #     identity = self.downsample(x)
            # return self.net(x) + identity
            
            return self.skip(x) + self.net(x)
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()
        print("in_channels", in_channels)
        cnn_layers = [
            torch.nn.Conv2d(in_channels,64, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        c1=64
        for _ in range(3):
           c2 = c1*2
           cnn_layers.append(self.Block(c1, c2, stride=2))
           c1=c2

          # Adjusted stride for pooling
        
        self.oneconv= (torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.globalavgpool=(torch.nn.AdaptiveAvgPool2d(1))

        # cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        # cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))

        self.network = torch.nn.Sequential(*cnn_layers)

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        #print(self.network)

        # TODO: implement


        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        
        x = self.network(x)
        x = self.oneconv(x)

        logits = self.globalavgpool(x)

        return logits.view(logits.size(0), -1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

from torch.nn.functional import relu
class EncoderBlock(torch.nn.Module):
  def __init__(
        self, n_input, n_output, stride=1
    ):

    self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )



class DecoderBlock(torch.nn.Module):
  def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        bilinear=False
    ):
    pass

class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        bilinear=False
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()
        # Encoder
        # -------
        # Encoder - Removed one layer
        # -------
        # input: 96x128x3
        self.e11 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # output: 96x128x32
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # output: 96x128x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 48x64x32

        # input: 48x64x32
        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # output: 48x64x64
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 48x64x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 24x32x64

        # input: 24x32x64
        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 24x32x128
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 24x32x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 12x16x128

        # input: 12x16x128 - REMOVED one encoder layer (e41, e42, pool4)

        # input: 12x16x128 - Bottom layer now has 128 channels
        self.e51 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 12x16x256  (Reduced channels)
        self.e52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 12x16x256

        # Decoder - Removed one layer
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # output: 24x32x128
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # input from cat: 24x32x(128+128)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 24x32x128

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # output: 48x64x64
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # input from cat: 48x64x(64+64)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 48x64x64

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # output: 96x128x32
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # input from cat: 96x128x(32+32)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # output: 96x128x32

        # Removed one decoder layer (upconv4, d41, d42)

        # Output layers - separated for classes and depth
        self.outconv_classes = nn.Conv2d(32, num_classes, kernel_size=1)  # output: 96x128xn_classes
        self.outconv_depth = nn.Conv2d(32, 1, kernel_size=1)  # output: 96x128xn_depths


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        #z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        # raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))

 # Encoder
         # Encoder
        x1 = self.e11(x)
        x1 = self.e12(x1)  # x1 shape = (B, 32, 96, 128)
        x_pool1 = self.pool1(x1)  # x_pool1 shape = (B, 32, 48, 64)

        x2 = self.e21(x_pool1)
        x2 = self.e22(x2)  # x2 shape = (B, 64, 48, 64)
        x_pool2 = self.pool2(x2)  # x_pool2 shape = (B, 64, 24, 32)

        x3 = self.e31(x_pool2)
        x3 = self.e32(x3)  # x3 shape = (B, 128, 24, 32)
        x_pool3 = self.pool3(x3)  # x_pool3 shape = (B, 128, 12, 16)

        # x4 = self.e41(x_pool3)  # REMOVED
        # x4 = self.e42(x4)      # REMOVED
        # x_pool4 = self.pool4(x4)  # REMOVED

        # x5 = self.e51(x_pool4)  # REMOVED
        # x5 = self.e52(x5)      # REMOVED

        x5 = self.e51(x_pool3) #input is now x_pool3 shape = (B, 128, 12, 16)
        x5 = self.e52(x5)  # x5 shape = (B, 256, 12, 16) #Reduced channels to 256

        # Decoder
        x = self.upconv1(x5)  # x shape = (B, 128, 24, 32)
        x = torch.cat([x, x3], dim=1)  # x shape = (B, 256, 24, 32)
        x = self.d11(x)
        x = self.d12(x)  # x shape = (B, 128, 24, 32)

        x = self.upconv2(x)  # x shape = (B, 64, 48, 64)
        x = torch.cat([x, x2], dim=1)  # x shape = (B, 128, 48, 64)
        x = self.d21(x)
        x = self.d22(x)  # x shape = (B, 64, 48, 64)

        x = self.upconv3(x)  # x shape = (B, 32, 96, 128)
        x = torch.cat([x, x1], dim=1)  # x shape = (B, 64, 96, 128)
        x = self.d31(x)
        x = self.d32(x)  # x shape = (B, 32, 96, 128)

        # x = self.upconv4(x)  # REMOVED
        # x = torch.cat([x, x1], dim=1)  # REMOVED
        # x = self.d41(x)  # REMOVED
        # x = self.d42(x)  # REMOVED

        # Output - now separate
        logits = self.outconv_classes(x)  # logits shape = (B, 3, 96, 128)
        depth = self.outconv_depth(x)  # depth shape = (B, 1, 96, 128)

        return logits, depth.squeeze(1)  # Return (B, 3, 96, 128) and (B, 96, 128)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()

