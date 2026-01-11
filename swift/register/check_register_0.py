import sys
from pprint import pprint

from swift.llm.model import MODEL_ARCH_MAPPING, MODEL_MAPPING
from swift.llm.template import get_template_meta

from rich.pretty import pprint
import swift.register.point_cloud_register  # noqa: F401


EXPECTED_ARCH = {
    'language_model': ['thinker.model', 'thinker.lm_head'],
    'vision_tower': ['thinker.audio_tower', 'thinker.visual', 'thinker.point_encoder'],
    'aligner': [
        'thinker.audio_tower.proj1',
        'thinker.audio_tower.proj2',
        'thinker.visual.merger',
        'thinker.visual.merger_list',
        'thinker.point_projector',
    ],
    'generator': ['talker', 'code2wav'],
}


def main() -> None:
    model_type = 'qwen3_omni'
    assert model_type in MODEL_MAPPING, f'Model type {model_type} not registered.'
    model_meta = MODEL_MAPPING[model_type]
    pprint(model_meta)




    template_meta = get_template_meta(model_type)
    pprint(template_meta)

    template_cls = template_meta.template_cls
    pprint(template_cls)




if __name__ == '__main__':
    main()
