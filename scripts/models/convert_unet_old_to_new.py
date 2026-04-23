# %%

import torch
from scripts.models.unet import UNet


# %%

def convert_unet_state_dict(old_state_dict: dict) -> dict:
    new_state_dict = {}

    for key, value in old_state_dict.items():
        if key.startswith("conv") \
                and (key.endswith(".weight") or key.endswith(".bias")) \
                and not "conv93" in key:
            # e.g. conv11.weight -> conv11.seq.0.weight
            parts = key.split(".")
            layer_name = parts[0]  # conv11
            param_name = parts[1]  # weight or bias

            new_key = f"{layer_name}.seq.0.{param_name}"
            new_state_dict[new_key] = value

        else:
            # everything else stays unchanged (pool, upconv, final conv, etc.)
            new_state_dict[key] = value

    return new_state_dict


old_sd = torch.load("models/20260420_221930_unet_cv_cos_150e_old/model.pt")
new_sd = convert_unet_state_dict(old_sd)

model = UNet(n_classes=18, norm="none")
model.load_state_dict(new_sd)
torch.save(model.state_dict(), "models/20260420_221930_unet_cv_cos_150e_new/model.pt")
