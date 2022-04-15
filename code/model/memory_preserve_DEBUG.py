import sys
import signal
import code.model.trainer as trainer
from importlib import reload

params={'batch_size':[100, 200, 256, 300, 400]}
folders={'batch_size':'batch size'}

options_cache, env_cache = trainer.setup()

for param in params.keys():
    for value in params[param]:
        temp_options = options_cache
        temp_options[param]=value
        temp_options['hp_type']=folders[param]
        temp_options['hp_level']=str(value)
        env_cache.batch_size=value
        env_cache.batcher.batch_size=value
        try:
            trainer.train(options_cache, env_cache)
        # except KeyboardInterrupt:
        #     print("interrupted")
        #     pass
        except Exception as e:
            print(e)
            print("Failure")
            pass
        # print("Press enter to run again")
        # input()
        reload(trainer)