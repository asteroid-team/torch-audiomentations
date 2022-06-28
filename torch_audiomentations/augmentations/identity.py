from typing import Optional
from torch import Tensor

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class Identity(BaseWaveformTransform):

    supported_modes = {"per_batch", "per_example", "per_channel"}
    supports_multichannel = True
    requires_sample_rate = False
    supports_target = True
    requires_target = False

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
