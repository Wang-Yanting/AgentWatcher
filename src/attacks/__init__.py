from .attack_heuristic import *
from .attack_none import *

ATTACKS = {
    "combined": combined,
    "ignore": ignore,
    "completion": completion,
    "character": character,
    "naive": naive,
    "none": no_attack,
}
