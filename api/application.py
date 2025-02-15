import sys

sys.path.append(sys.path[0]+'\\..')



import src as RecSys

help(RecSys)

#
# Note: Scripts are needed to be moved.
#



#
# Train Example Script
#

df_s, df_o, df_a = RecSys.data.load(sys.path[0] + '\\..\\src\\data\\raw')

dvp = RecSys.core.pipeline.DataValidationPipeline(exclude = ['birth', 'rate', 'timestamp'], time = ['birth', 'timestamp'])
df_clean_s = dvp.process(df_s)
df_clean_o = dvp.process(df_o)
df_clean_a = dvp.process(df_a)

mep = RecSys.core.pipeline.ModelEmbeddingPipeline(df_clean_s,df_clean_o,df_clean_a, RecSys.core.config.Config)
dc, dl = mep.process()
dc['batch_size'] = RecSys.core.config.Config['batch_size']




#
# Inference Example Script
#