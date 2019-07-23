# TODO: sholl we use CSV to save our data?
import csv
import tushare as ts
import opts
from torch.utils.data import DataLoader, Dataset

class DownloadData(object):
    """
    Arguments:
        data_source: list, d 
    """
    def __init__(self, opt):
        self.data_source = opt.data_source

    def get_k_data(self):
        # data = ts.
        pass

    def __get_code_list__(self):
        if self.data_source == 'zz500':
            return [i for i in ts.get_zz500s()['code']]
        elif self.data_source == 'sz50':
            return [i for i in ts.get_sz50s()['code']]
        elif self.data_source == 'hs300':
            return [i for i in ts.get_hs300s()['code']]
        # TODO: we can add other data source






class StockDataset(Dataset):
    def __init__(self, opt):
        """
        mode: train or backtest
        """
        self.mode = opt.mode
        self.data_source = opt.data_source
        self

    def save_vanilla(self):
        pass


    def __get_item__(self, index):
        """ Get specific original data from DataLoader
        """
        pass
    
    def __get_date__(self, index):
        """
        """
        pass

    def __get_price__(self, index):
        pass

    def __get_code__(self, index):
        pass
    
    def __get_path__(self):
        """ 
        """
        pass
        return path 
        

if __name__ == "__main__":
    opt = opts.parse_opt()
    # Test case 1

    data = DownloadData(opt)
    l = data.__get_code_list__()
    print(l)
    