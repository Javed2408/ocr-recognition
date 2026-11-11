"""Font families used to imitate handwriting and printed cheque text.

Families (not single files) are split between train/val/test so that no
handwriting style seen in training appears in evaluation.
"""
import os
from functools import lru_cache

from PIL import ImageFont

from ..config import FONTS_DIR

HANDWRITING_FAMILIES = {
    "train": {
        "inkfree": ["Inkfree.ttf"], "segoe_print": ["segoepr.ttf", "segoeprb.ttf"],
        "lucida_hand": ["LHANDW.TTF"], "kristen": ["ITCKRIST.TTF"], "comic": ["comic.ttf", "comici.ttf"],
        "mistral": ["MISTRAL.TTF"], "pristina": ["PRISTINA.TTF"], "juice": ["JUICE___.TTF"],
        "viner": ["VINERITC.TTF"], "script_mt": ["SCRIPTBL.TTF"], "rage": ["RAGE.TTF"],
        "kunstler": ["KUNSTLER.TTF"], "palace": ["PALSCRI.TTF"], "french_script": ["FRSCRIPT.TTF"],
    },
    "val": {"bradley": ["BRADHITC.TTF"], "freestyle": ["FREESCPT.TTF"], "gigi": ["GIGI.TTF"]},
    "test": {"segoe_script": ["segoesc.ttf", "segoescb.ttf"], "mvboli": ["mvboli.ttf"],
             "gabriola": ["Gabriola.ttf"], "vladimir": ["VLADIMIR.TTF"]},
}
# Scripts too thin/decorative to render small digits legibly are used for words only
WORDS_ONLY = {"kunstler", "palace", "french_script", "vladimir", "rage"}

PRINTED_FAMILIES = {
    "train": ["arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "calibri.ttf", "calibrib.ttf",
              "cambriab.ttf", "consola.ttf", "cour.ttf", "courbd.ttf", "verdana.ttf", "verdanab.ttf",
              "trebuc.ttf", "trebucbd.ttf", "segoeui.ttf", "segoeuib.ttf", "corbel.ttf", "corbelb.ttf",
              "constan.ttf", "lucon.ttf", "micross.ttf", "ARIALN.TTF", "ARIALNB.TTF"],
    "val": ["georgia.ttf", "georgiab.ttf"],
    "test": ["Candara.ttf", "Candarab.ttf", "tahoma.ttf", "tahomabd.ttf"],
}
RUPEE_FONT = "Nirmala.ttf"          # has the U+20B9 rupee sign
MICR_FONT = "OCRAEXT.TTF"           # OCR-A, a stand-in for the E-13B MICR font


def _exists(name):
    return os.path.isfile(os.path.join(FONTS_DIR, name))


def handwriting_families(split, digits=False):
    fams = {k: [f for f in v if _exists(f)] for k, v in HANDWRITING_FAMILIES[split].items()}
    fams = {k: v for k, v in fams.items() if v and not (digits and k in WORDS_ONLY)}
    if not fams:
        raise RuntimeError(f"no handwriting fonts found in {FONTS_DIR} for split {split!r}")
    return fams


def printed_fonts(split):
    fonts = [f for f in PRINTED_FAMILIES[split] if _exists(f)]
    if not fonts:
        raise RuntimeError(f"no printed fonts found in {FONTS_DIR} for split {split!r}")
    return fonts


@lru_cache(maxsize=512)
def load_font(name, size):
    return ImageFont.truetype(os.path.join(FONTS_DIR, name), int(size))
