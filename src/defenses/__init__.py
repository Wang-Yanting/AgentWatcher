from .datasentinel import *
from .promptguard import *
from .promptarmor import *
from .piguard import *
from .gptsafeguard import *
from .defense_none import *
from .agentwatcher import *

DEFENSES = {
    "datasentinel": datasentinel,
    "promptguard": promptguard,
    "promptarmor": promptarmor,
    "piguard": piguard,
    "gptsafeguard": gptsafeguard,
    "agentwatcher": agentwatcher,
    "none": no_defense,
}

DEFENSES_BATCH = {
    "promptarmor": promptarmor_batch,
    "datasentinel": datasentinel_batch,
    "promptguard": promptguard_batch,
    "piguard": piguard_batch,
    "gptsafeguard": gptsafeguard_batch,
    "agentwatcher": agentwatcher_batch,
    "none": no_defense_batch,
}


class DefenseWrapper:
    """Wrapper class that provides execute() method for defense functions."""
    def __init__(self, defense_func):
        self.defense_func = defense_func
    
    def execute(self, target_inst, context, **kwargs):
        """Execute the defense and return result dict."""
        return self.defense_func(target_inst=target_inst, context=context, **kwargs)


def get_defense(defense_name: str):
    """
    Get a defense instance by name.
    
    Args:
        defense_name: Name of the defense (e.g., 'promptguard', 'agentwatcher')
        
    Returns:
        DefenseWrapper instance with execute() method
    """
    defense_name = defense_name.lower()
    if defense_name not in DEFENSES:
        raise ValueError(f"Unknown defense: {defense_name}. Available: {list(DEFENSES.keys())}")
    
    return DefenseWrapper(DEFENSES[defense_name])
