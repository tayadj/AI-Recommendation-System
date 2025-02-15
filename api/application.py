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
dvp = RecSys.core.pipeline.DataValidationPipeline(exclude = ['birth', 'rate', 'timestamp'])
df_clean_s = dvp.process(df_s)
df_clean_o = dvp.process(df_o)
df_clean_a = dvp.process(df_a)


#
# Inference Example Script
#