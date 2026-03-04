from algs.mano import MaNo
from algs.mano_attack import (
    MaNoMaxAttack, MaNoMinAttack,
    MaNoMaxAttackFast, MaNoMinAttackFast,
)


def create_alg(alg_name, val_loader, device, args, base_model=None):
    alg_dict = {
        'mano': MaNo,
        'mano_max': MaNoMaxAttack,
        'mano_min': MaNoMinAttack,
        'mano_max_fast': MaNoMaxAttackFast,
        'mano_min_fast': MaNoMinAttackFast,
    }
    model = alg_dict[alg_name](val_loader, device, args, base_model=base_model)
    return model
