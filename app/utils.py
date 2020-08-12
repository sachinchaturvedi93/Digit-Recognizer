import numpy as np 
import pandas as pd 
from fastai import *
from fastai.vision import *

class CustomImageList(ImageList):
    def open(self, fn):
        img = fn.reshape(28,28)
        img = np.stack((img,)*3, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))
    
    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res
    
    @classmethod
    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res
