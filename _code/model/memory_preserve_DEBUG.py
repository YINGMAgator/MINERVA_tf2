import sys
import os
import signal
import code.model.trainer as trainer
from importlib import reload

params={'lambda':[0.02,0.2,2],'beta':[0.0002,0.002,0.02,0.2,2],'batch_size':[100, 200, 256, 300, 400]}
folders={'learning_rate':'learning rate_advanced','lambda':'lambda_advanced','beta':'beta_advanced','batch_size':'batch size'}

options_cache, env_cache = trainer.setup()
trainer.train(options_cache, env_cache)
# for param in params.keys():
#     for value in params[param]:
#         temp_options = options_cache
#         temp_options[param]=value
#         temp_options['hp_type']=folders[param]
#         temp_options['hp_level']=str(value)+"_final"

#         if param == "batch_size":
#             env_cache.batch_size=value
#             env_cache.batcher.batch_size=value
#         trainer.train(options_cache, env_cache)
#         try:
#             
#         # except KeyboardInterrupt:
#         #     print("interrupted")
#         #     pass
#         except Exception as e:
#             exc_type, exc_obj, exc_tb = sys.exc_info()
#             fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#             print(exc_type, fname, exc_tb.tb_lineno)
#             print(e)
#             print("Failure")
#             pass
#         # print("Press enter to run again")
#         # input()
#         reload(trainer)