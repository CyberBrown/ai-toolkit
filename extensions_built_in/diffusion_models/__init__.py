from .chroma import ChromaModel, ChromaRadianceModel
try:
    from .hidream import HidreamModel, HidreamE1Model
except Exception as _e:  # transformers-pin mismatch on GB10
    print(f"[diffusion_models] skipping: {_e}")
    HidreamModel = HidreamE1Model = None
from .f_light import FLiteModel
from .omnigen2 import OmniGen2Model
from .flux_kontext import FluxKontextModel
from .wan22 import Wan225bModel, Wan2214bModel, Wan2214bI2VModel
from .qwen_image import QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel
from .qwen_image_2 import QwenImage2Model
from .ming_image import MingImageModel
from .flux2 import Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel
from .z_image import ZImageModel
try:
    from .ltx2 import LTX2Model, LTX23Model, LTX25Model
except Exception as _e:  # needs newer diffusers than the GB10 pin
    print(f"[diffusion_models] skipping: {_e}")
    LTX2Model = LTX23Model = LTX25Model = None
from .zeta_chroma import ZetaChromaModel
from .ernie_image import ErnieImageModel
from .nucleus_image import NucleusImageModel
try:
    from .hidream.hidream_o1_model import HidreamO1Model
except Exception as _e:  # transformers-pin mismatch on GB10
    print(f"[diffusion_models] skipping: {_e}")
    HidreamO1Model = None
from .z_image.z_image_l2p_model import ZImageL2PModel
from .anima import AnimaModel
from .ideogram4 import Ideogram4Model
from .prx_pixel_t2i import PRXPixelT2IModel
from .krea2 import Krea2Model
from .iris3b import Iris3BModel
from .boogu_image import BooguImageModel, BooguImageEditModel
from .mageflow import MageFlowModel, MageFlowEditModel
from .minimax_h3 import MinimaxH3Model, MinimaxH3Ref2VAModel, MinimaxH3FastModel, MinimaxH3OmniModel
from .flux3 import Flux3Model
from .moss_image import MossImageModel
from .fibo2 import Fibo2Model

AI_TOOLKIT_MODELS = [
    # put a list of models here
    ChromaModel,
    ChromaRadianceModel,
    HidreamModel,
    HidreamE1Model,
    FLiteModel,
    OmniGen2Model,
    FluxKontextModel,
    Wan225bModel,
    Wan2214bI2VModel,
    Wan2214bModel,
    QwenImageModel,
    QwenImageEditModel,
    QwenImageEditPlusModel,
    QwenImage2Model,
    MingImageModel,
    Flux2Model,
    ZImageModel,
    LTX2Model,
    LTX23Model,
    LTX25Model,
    Flux2Klein4BModel,
    Flux2Klein9BModel,
    ZetaChromaModel,
    ErnieImageModel,
    NucleusImageModel,
    HidreamO1Model,
    ZImageL2PModel,
    AnimaModel,
    Ideogram4Model,
    PRXPixelT2IModel,
    Krea2Model,
    Iris3BModel,
    BooguImageModel,
    BooguImageEditModel,
    MageFlowModel,
    MageFlowEditModel,
    MinimaxH3Model,
    MinimaxH3Ref2VAModel,
    MinimaxH3FastModel,
    MinimaxH3OmniModel,
    Flux3Model,
    MossImageModel,
    Fibo2Model,
]

AI_TOOLKIT_MODELS = [m for m in AI_TOOLKIT_MODELS if m is not None]
