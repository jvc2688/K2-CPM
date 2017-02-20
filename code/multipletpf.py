import numpy as np

import tpfdata


class MultipleTpf(object):
    """keeps a collection of TPF files"""

    def __init__(self):
        self._tpfs = set()
        self._epic_ids = set()
        self._predictor_epoch_mask = None
        self._campaign = None

    def add_tpf_data(self, tpf_data):
        """add one more instance of TpfData"""
        if not isinstance(tpf_data, tpfdata.TpfData):
            raise ValueError('Ooops... MultipleTpf.add_tpf_data() requires input that is an instance of TpfData class')
        if tpf_data.epic_id in self._epic_ids:
            return
        if self._campaign is None:
            self._campaign = tpf_data.campaign
        else:
            if self._campaign != tpf_data.campaign:
                msg = 'MultipleTpf.add_tpf_data() cannot add data from a different campaign ({:} and {:})'
                raise ValueError(msg.format(self._campaign, tpf_data.campaign))
            
        self._tpfs.add(tpf_data)
        self._epic_ids.add(tpf_data.epic_id)
        if self._predictor_epoch_mask is None:
            self._predictor_epoch_mask = np.ones_like(tpf_data.epoch_mask, dtype=bool)
        self._predictor_epoch_mask &= tpf_data.epoch_mask

    @property
    def predictor_epoch_mask(self):
        """predictor epoch mask"""
        return self._predictor_epoch_mask

    def add_tpf_data_from_epic_list(self, epic_id_list, campaign):
        """for each epic_id in the list, construct TPF object and add it to the set"""
        for epic_id in epic_id_list:
            new_tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
            self.add_tpf_data(new_tpf)
            
    def add_tpf_data_from_epic_list_in_file(self, epic_id_list_file, campaign):
        """read data from file and apply add_tpf_data_from_epic_list()"""
        with open(epic_id_list_file) as list_file:
            epic_id_list = [int(line) for line in list_file.readlines()]
        self.add_tpf_data_from_epic_list(epic_id_list=epic_id_list, campaign=campaign)


if __name__ == '__main__':
    directory = 'tpf/'
    campaign = 92 
    tpfs_epic_list_file = "../test/output/1-epic.dat"
    out_file = '1-predictor_epoch_mask.dat'
    
    tpfdata.TpfData.directory = directory
    
    tpf_set = MultipleTpf()
    tpf_set.add_tpf_data_from_epic_list_in_file(tpfs_epic_list_file, campaign=campaign)
    predictor_epoch_mask = tpf_set.predictor_epoch_mask

    np.savetxt(out_file, predictor_epoch_mask, fmt='%r')
    