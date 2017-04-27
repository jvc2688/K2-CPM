import numpy as np
from bisect import bisect

#import tpfdata
from K2CPM import tpfdata


class MultipleTpf(object):
    """keeps a collection of TPF files"""

    def __init__(self):
        self._tpfs = [] # This list has TpfData instances and the order corresponds to self._epic_ids.
        self._epic_ids = [] # This is sorted list and all elements are of string type.
        self._predictor_epoch_mask = None
        self._campaign = None
        
        self._get_rows_columns_epics = None
        self._get_fluxes_epics = None
        self._get_median_fluxes_epics = None

    def add_tpf_data(self, tpf_data):
        """add one more instance of TpfData"""
        if not isinstance(tpf_data, tpfdata.TpfData):
            msg = 'Ooops... MultipleTpf.add_tpf_data() requires input that is an instance of TpfData class'
            raise ValueError(msg)
        if tpf_data.epic_id in self._epic_ids:
            return
        if self._campaign is None:
            self._campaign = tpf_data.campaign
        else:
            if self._campaign != tpf_data.campaign:
                msg = 'MultipleTpf.add_tpf_data() cannot add data from a different campaign ({:} and {:})'
                raise ValueError(msg.format(self._campaign, tpf_data.campaign))
        
        index = bisect(self._epic_ids, tpf_data.epic_id)
        self._tpfs.insert(index, tpf_data)
        self._epic_ids.insert(index, tpf_data.epic_id)
        
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
            if epic_id in self._epic_ids:
                continue
            new_tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
            self.add_tpf_data(new_tpf)
            
    def add_tpf_data_from_epic_list_in_file(self, epic_id_list_file, campaign):
        """read data from file and apply add_tpf_data_from_epic_list()"""
        with open(epic_id_list_file) as list_file:
            epic_id_list = [int(line) for line in list_file.readlines()]
        self.add_tpf_data_from_epic_list(epic_id_list=epic_id_list, campaign=campaign)

    def _limit_epic_ids_to_list(self, epic_list):
        """limit self._epic_ids to ones in epic_list"""
        out = []
        for epic in self._epic_ids:
            if epic in epic_list:
                out.append(epic)
        return out

    def get_rows_columns(self, epics_to_include):
        """get concatenated rows and columns for selected epics"""
        get_rows_columns_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_rows_columns_epics == self._get_rows_columns_epics:
            print("XXXXX SAME XXXXX")
            return (self._get_rows_columns_rows, self._get_rows_columns_columns)
        print("XXXXX different XXXXXX")
        self._get_rows_columns_epics = get_rows_columns_epics

        rows = []
        columns = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            rows.append(self._tpfs[i].rows)
            columns.append(self._tpfs[i].columns)
        self._get_rows_columns_rows = np.concatenate(rows, axis=0).astype(int)
        self._get_rows_columns_columns = np.concatenate(columns, axis=0).astype(int)
        return (self._get_rows_columns_rows, self._get_rows_columns_columns)

    def get_fluxes(self, epics_to_include):
        """get concatenated fluxes for selected epics"""
        get_fluxes_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_fluxes_epics == self._get_fluxes_epics:
            return self._get_fluxes
        self._get_fluxes_epics = get_fluxes_epics

        flux = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            flux.append(self._tpfs[i].flux)
        self._get_fluxes = np.concatenate(flux, axis=1)
        return self._get_fluxes

    def get_median_fluxes(self, epics_to_include):
        """get concatenated median fluxes for selected epics"""
        get_median_fluxes_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_median_fluxes_epics == self._get_median_fluxes_epics:
            return self._get_median_fluxes
        self._get_median_fluxes_epics = get_median_fluxes_epics

        median_flux = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            median_flux.append(self._tpfs[i].median)
        self._get_median_fluxes = np.concatenate(median_flux, axis=0)
        return self._get_median_fluxes


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
    
