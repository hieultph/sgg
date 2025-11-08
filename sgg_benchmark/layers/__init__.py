from .batch_norm import FrozenBatchNorm2d
from .misc import Conv2d
from .misc import DFConv2d
from .misc import ConvTranspose2d
from .misc import BatchNorm2d
from .misc import interpolate
from .misc import MLP
from .misc import fusion_func
from torchvision.ops import nms
from torchvision.ops import RoIAlign as ROIAlign
from torchvision.ops import roi_align, roi_pool
from torchvision.ops import RoIPool as ROIPool
from .entropy_loss import entropy_loss
from .kl_div_loss import kl_div_loss
from .smooth_l1_loss import smooth_l1_loss
from .sigmoid_focal_loss import SigmoidFocalLoss
from .label_smoothing_loss import Label_Smoothing_Regression

# Try to import deformable convolution functions
try:
    from .dcn.deform_conv_func import deform_conv, modulated_deform_conv
    from .dcn.deform_conv_module import DeformConv, ModulatedDeformConv, ModulatedDeformConvPack
    from .dcn.deform_pool_func import deform_roi_pooling
    from .dcn.deform_pool_module import DeformRoIPooling, DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack
    DCN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deformable convolution not available: {e}")
    print("Using fallback implementations - some models may not work correctly")
    # Create dummy functions as fallbacks
    def deform_conv(*args, **kwargs):
        raise NotImplementedError("Deformable convolution not available - CUDA extensions not compiled")
    def modulated_deform_conv(*args, **kwargs):
        raise NotImplementedError("Modulated deformable convolution not available - CUDA extensions not compiled")
    class DeformConv:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DeformConv not available - CUDA extensions not compiled")
    class ModulatedDeformConv:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ModulatedDeformConv not available - CUDA extensions not compiled") 
    class ModulatedDeformConvPack:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ModulatedDeformConvPack not available - CUDA extensions not compiled")
    def deform_roi_pooling(*args, **kwargs):
        raise NotImplementedError("Deform RoI pooling not available - CUDA extensions not compiled")
    class DeformRoIPooling:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DeformRoIPooling not available - CUDA extensions not compiled")
    class DeformRoIPoolingPack:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DeformRoIPoolingPack not available - CUDA extensions not compiled")
    class ModulatedDeformRoIPoolingPack:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ModulatedDeformRoIPoolingPack not available - CUDA extensions not compiled")
    DCN_AVAILABLE = False


__all__ = [
    "nms",
    "roi_align",
    "ROIAlign",
    "roi_pool",
    "ROIPool",
    "smooth_l1_loss",
    "entropy_loss",
    "kl_div_loss",
    "Conv2d",
    "DFConv2d",
    "ConvTranspose2d",
    "interpolate",
    "BatchNorm2d",
    "FrozenBatchNorm2d",
    "SigmoidFocalLoss",
    "Label_Smoothing_Regression",
    'deform_conv',
    'modulated_deform_conv',
    'DeformConv',
    'ModulatedDeformConv',
    'ModulatedDeformConvPack',
    'deform_roi_pooling',
    'DeformRoIPooling',
    'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack',
]

