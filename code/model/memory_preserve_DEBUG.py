import sys
import signal
import code.model.trainer as trainer
from importlib import reload

env_cache = None
options_cache = None
while True:
    if not env_cache or not options_cache:
        options_cache, env_cache = trainer.setup()
    try:
        trainer.train(options_cache, env_cache)
    except KeyboardInterrupt:
        print("interrupted")
        pass
    except Exception as e:
        print(e)
        pass
    print("Press enter to run again")
    input()
    reload(trainer)
