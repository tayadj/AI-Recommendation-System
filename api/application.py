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

df_s, df_o, df_a = RecSys.data.load('./src/data/raw')
dvp = RecSys.core.pipeline.DataValidationPipeline()
df_clean_s = dvp.process(df_s)


#
# Inference Example Script
#