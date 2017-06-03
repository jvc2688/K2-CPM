import numpy as np
from bisect import bisect

#import tpfdata
from K2CPM import tpfdata, hugetpf


class MultipleTpf(object):
    """keeps a collection of TPF files"""

    def __init__(self, n_remove_huge=None, campaign=None):
        self._tpfs = [] # This list has TpfData instances and the order corresponds to self._epic_ids.
        self._epic_ids = [] # This is sorted list and all elements are of string type.
        self._predictor_epoch_mask = None
        self._campaign = campaign
        
        self._get_rows_columns_epics = None
        self._get_fluxes_epics = None
        self._get_median_fluxes_epics = None
        
        self._n_remove_huge = n_remove_huge
        self._huge_tpf = None

    def add_tpf_data(self, tpf_data):
        """add one more instance of TpfData"""
        if not isinstance(tpf_data, tpfdata.TpfData):
            msg = 'Ooops... MultipleTpf.add_tpf_data() requires input that is an instance of TpfData class'
            raise ValueError(msg)
        epic_id = str(tpf_data.epic_id)
        if epic_id in self._epic_ids:
            return
        if self._campaign is None:
            self._campaign = tpf_data.campaign
            if self._n_remove_huge is None:
                self._huge_tpf = hugetpf.HugeTpf(campaign=self._campaign)
            else:
                self._huge_tpf = hugetpf.HugeTpf(campaign=self._campaign, n_huge=self._n_remove_huge)
        else:
            if self._campaign != tpf_data.campaign:
                msg = 'MultipleTpf.add_tpf_data() cannot add data from a different campaign ({:} and {:})'
                raise ValueError(msg.format(self._campaign, tpf_data.campaign))
        
        index = bisect(self._epic_ids, epic_id)
        self._tpfs.insert(index, tpf_data)
        self._epic_ids.insert(index, epic_id)
        
        if self._predictor_epoch_mask is None:
            self._predictor_epoch_mask = np.ones_like(tpf_data.epoch_mask, dtype=bool)
        self._predictor_epoch_mask &= tpf_data.epoch_mask

    def tpf_for_epic_id(self, epic_id, add_if_not_present=True):
        """returns an instance of TpfData corresponding to given epic_id"""
        epic_id = str(epic_id)
        if epic_id not in self._epic_ids:
            if not add_if_not_present:
                raise ValueError('EPIC {:} not in the MultipleTpf instance'.format(epic_id))
            self.add_tpf_data_from_epic_list([epic_id])
            
        index = self._epic_ids.index(epic_id)
        return self._tpfs[index]   

    @property
    def predictor_epoch_mask(self):
        """predictor epoch mask"""
        return self._predictor_epoch_mask

    def add_tpf_data_from_epic_list(self, epic_id_list, campaign=None):
        """for each epic_id in the list, construct TPF object and add it to the set"""
        if campaign is None:
            if self._campaign is None:
                raise ValueError('MultipleTpf - campaign not known')
            campaign = self._campaign
        for epic_id in epic_id_list:
            if str(epic_id) in self._epic_ids:
                continue
            if str(epic_id) in self._huge_tpf.huge_ids:
                continue
                # This way we skip huge TPF files, though they can still be added via self.add_tpf_data().
            new_tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
            self.add_tpf_data(new_tpf)
            
    def add_tpf_data_from_epic_list_in_file(self, epic_id_list_file, campaign=None):
        """read data from file and apply add_tpf_data_from_epic_list()"""
        if campaign is None:
            if self._campaign is None:
                raise ValueError('MultipleTpf() - campaign not known')
            campaign = self._campaign        
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
            return (self._get_rows_columns_rows, self._get_rows_columns_columns)
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
