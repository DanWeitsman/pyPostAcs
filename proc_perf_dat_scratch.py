
import numpy as np
import os
import re
import h5py

#%%
#    prefix of the path to a Full.txt
dir =  '/Users/danielweitsman/Desktop/fb77'

# #   opens Full.txt file
# data =  open(os.path.join(dir,'Full.txt'),'r').read()
# #   number of columns in Full.txt file
# col = 27
# #   splits the Full.txt file with \t and \n as delimiters
# data_split = re.split('\t|\n',data)
# #   reshapes a single-dimensional list into the right dimensional array
# data_split = np.reshape(data_split[:-1],(int(len(data_split[:-1])/col),col))
# #   extracts header of each data set in Full.txt
# header = data_split[0]
# #   changes numerical data from type str to float64
# data_split = data_split[1:].astype(float)

#%%
#   opens an h5 file
with h5py.File(os.path.join(dir,'acs_data.h5'),'a') as f_h5:
    #   opens and reads contents of Full.txt file
    with open(os.path.join(dir, 'Full.txt'), 'r') as f_txt:
        data = f_txt.read()
    #   number of columns in Full.txt file
    col = 27
    #   splits the Full.txt file with \t and \n as delimiters
    data_split = re.split('\t|\n', data)
    #   reshapes a single-dimensional list into the right dimensional array
    data_split = np.reshape(data_split[:-1], (int(len(data_split[:-1]) / col), col))
    #   extracts header of each data set in Full.txt
    header = data_split[0]
    #   changes numerical data from type str to float64
    data_split = data_split[1:].astype(float)
    #   loops through each dataset contained in Full.txt
    for i,dat in enumerate(data_split.transpose()):
        #   writes data to a new dataset titled with each header
        f_h5.create_dataset(header[i], data = dat, shape=np.shape(dat))


