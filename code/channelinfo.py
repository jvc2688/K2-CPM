
import requests
import json
import urllib


class ChannelInfo(object):
    """basic info on given K2 channel"""
    def __init__(self, campaign, channel):
        if campaign != 91 and campaign != 92:
            raise ValueError('ChannelInfo() accepts only campaigns 91 and 92 at this point') 
        self.campaign = campaign
        self.channel = channel

    def epic_ids_near_ra_dec(self, ra, dec, minimum_n=6, except_largest=2):
        """finds at least minimum_n EPIC ids closest to given ra/dec; up to 4 largest files are removed in order not to read huge files with almost no data"""
        if except_largest == 0:
            remove = set()
        elif except_largest == 1:
            remove = set(200070438)
        elif except_largest == 2:
            remove = set([200070438, 200070874])
        elif except_largest == 3:
            remove = set([200070438, 200070874, 200069673])
        elif except_largest == 4:
            remove = set([200070438, 200070874, 200069673, 200071158])
        else:
            raise ValueError('epic_ids_near_ra_dec() argument except_largest has to be between 0 and 4')

        url = 'https://archive.stsci.edu/k2/data_search/search.php'
        radius = 0 # This is in arc min most probably.
        epic_list = []
        while len(epic_list) < minimum_n: # TO BE DONE: Change it to calls of wcsfromtpf.py functions.
            radius += 6
            form_data = {"ra": ra, "dec": dec, "radius":radius, "sci_campaign":self.campaign, 'sci_channel':self.channel, "outputformat": "JSON", "action": "Search" }
            request = requests.post(url, data=form_data)
            result= json.loads(request.text)
            epic_set_raw = set([int(res['K2 ID']) for res in result])
            epic_list = list(epic_set_raw-remove)
        return epic_list

