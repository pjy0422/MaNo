from algs.mano import MaNo


def create_alg(alg_name, val_loader, device, args, base_model=None):
    alg_dict = {

        'mano': MaNo,

    }
    model = alg_dict[alg_name](val_loader, device, args, base_model=base_model)
    return model
