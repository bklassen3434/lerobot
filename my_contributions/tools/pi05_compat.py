"""Compatibility shim so `lerobot/pi05_base` can be loaded by this checkout.

The published pi05_base checkpoint stores a preprocessor pipeline whose steps are named
`relative_actions_processor` / `absolute_actions_processor`, but this version of LeRobot
registers the same classes as `delta_actions_processor` / `absolute_actions_processor`.
Loading the checkpoint therefore dies with:

    KeyError: "Processor step 'relative_actions_processor' not found in registry"

Registering the missing alias against the identical class fixes the load without touching
upstream source. Import this module before `lerobot_train.main()`:

    python -c "import my_contributions.tools.pi05_compat, lerobot.scripts.lerobot_train as t; t.main()" <args>
"""

import logging

from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)

ALIASES = {
    "relative_actions_processor": RelativeActionsProcessorStep,
    "absolute_actions_processor": AbsoluteActionsProcessorStep,
}

for name, step_class in ALIASES.items():
    try:
        ProcessorStepRegistry.get(name)
    except KeyError:
        # Register directly rather than via the decorator: the decorator also rewrites
        # `_registry_name`, which would change how *newly saved* pipelines are serialized.
        ProcessorStepRegistry._registry[name] = step_class
        logging.info("pi05_compat: registered alias '%s' -> %s", name, step_class.__name__)
