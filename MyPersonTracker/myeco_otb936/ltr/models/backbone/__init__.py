from .resnet import resnet18, resnet50, resnet101, resnet_baby
from .resnet18_vggm import resnet18_vggmconv1
try:
    from .swin_transformer_flex import swin_base384_flex
except ImportError:
    swin_base384_flex = None
