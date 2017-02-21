
import requests
import json
import urllib

import hugetpf


class ChannelInfo(object):
    """basic info on given K2 channel"""
    def __init__(self, campaign, channel):
        if campaign != 91 and campaign != 92:
            raise ValueError('ChannelInfo() accepts only campaigns 91 and 92 at this point') 
        self.campaign = campaign
        self.channel = channel
        self.radius = None
        self.epic_list = None

    def epic_ids_near_ra_dec_fixed_radius(self, ra, dec, except_largest=2):
        """queries for EPID ids that are within radius from given ra/dec"""
        # TO BE DONE: Change it to calls of wcsfromtpf.py functions.
        # Variable radius is in arc min most probably.
        url = 'https://archive.stsci.edu/k2/data_search/search.php'
        form_data = {"ra": ra, "dec": dec, "radius": self.radius, "sci_campaign": self.campaign, 'sci_channel': self.channel, "outputformat": "JSON", "action": "Search"}
        request = requests.post(url, data=form_data)
        result = json.loads(request.text)
        epic_set_raw = set([int(res['K2 ID']) for res in result])
        huge_tpf = hugetpf.HugeTpf(n_huge=except_largest, campaign=self.campaign)
        self.epic_list = list(epic_set_raw - huge_tpf.huge_ids)
        return len(self.epic_list)

    def epic_ids_near_ra_dec(self, ra, dec, minimum_n=6, except_largest=2, step_radius=6):
        """finds at least minimum_n EPIC ids closest to given ra/dec; up to n=except_largest largest files are removed in order not to read huge files with almost no data"""
        if self.radius is None:
            self.radius = 0 
        if self.epic_list is None:
            self.epic_list = []
        while True: # TO BE DONE: Change it to calls of wcsfromtpf.py functions.
            self.radius += step_radius
            current_n = self.epic_ids_near_ra_dec_fixed_radius(ra=ra, dec=dec, except_largest=except_largest)
            if current_n >= minimum_n:
                break
        return len(self.epic_list)

