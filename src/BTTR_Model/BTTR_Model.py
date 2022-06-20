# from pytorch_lightning import Trainer
# from bttr.datamodule import CROHMEDatamodule

from .bttr.lit_bttr import LitBTTR

# from PIL import Image
from torchvision.transforms import ToTensor
import os


def bttr_model_predict(image):
    # test_year = "2014"
    ckp_path = os.path.join(os.getcwd() + "/src/BTTR_Model/epoch=233-step=87983-val_ExpRate=0.5492.ckpt")

    # trainer = Trainer(logger=False, gpus=0)
    #
    # dm = CROHMEDatamodule(test_year=test_year)

    model = LitBTTR.load_from_checkpoint(ckp_path)

    # trainer.test(model, datamodule=dm)
    # directory = "2014"  # "2014"
    # ext = ".bmp"

    img = ToTensor()(image)
    hyp = model.beam_search(img)

    return hyp
