

class MultipleTpf(object):
    """keeps a collection of TPF files"""

    def __init__(self):
        self._tpfs = set()
        self._pixel_column_list = []
        self._pixel_row_list = []
        self._median_list = []
        self._pixel_flux_list = []
        self._predictor_epoch_mask = None

    def add_tpf_data(self, tpf_data):
        """add one more instance of TpfData"""
        self._tpfs.add(tpf_data)
        self._pixel_column_list.append(tpf_data.pixel_column)
        self._pixel_row_list.append(tpf_data.pixel_row)
        self._median_list.append(tpf_data.median)
        self._pixel_flux_list.append(tpf_data.flux)
        if self._predictor_epoch_mask is None:
            self._predictor_epoch_mask = np.ones_like(tpf_data.epoch_mask, dtype=bool)
        self._predictor_epoch_mask &= tpf_data.epoch_mask

    @property
    def pixel_column(self):
        """table with all concatenated pixel columns"""
        return np.concatenate(self._pixel_column_list, axis=0).astype(int)

    @property
    def pixel_row(self):
        """table with all concatenated pixel rows"""
        return np.concatenate(self._pixel_row_list, axis=0).astype(int)

    @property
    def pixel_median(self):
        """table with all concatenated median values"""
        return np.concatenate(self._median_list, axis=0)

    @property
    def pixel_flux(self):
        """table with all concatenated pixel flux values"""
        return np.concatenate(self._pixel_flux_list, axis=1)

    @property
    def predictor_epoch_mask(self):
        """predictor epoch mask"""
        return self._predictor_epoch_mask

