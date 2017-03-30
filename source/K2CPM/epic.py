from __future__ import print_function

import requests
import json
import urllib
import os
import sys


def get_tpfs(ra,dec,r,camp, channel):
    url = 'https://archive.stsci.edu/k2/data_search/search.php'
    data = {"ra": ra, "dec": dec, "radius":r, "sci_campaign":camp, 'sci_channel':channel, "outputformat": "JSON", "action": "Search" }
    r = requests.post(url, data=data)
    result= json.loads(r.text)
    r_list = [int(d['K2 ID']) for d in result]
    return list(set(r_list))

def file_name_for_epic(epic_id, campaign):
    """give name of file for given epic_id and campaign (91 or 92)"""
    return 'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(epic_id, campaign)
    
def path_for_epic(directory, epic_id, campaign):
    """return path to file with TPF file of given EPIC ID and campaign"""
    file_name = file_name_for_epic(epic_id, campaign)
    return os.path.join(directory, file_name)

def load_tpf(ID, camp, dire):
    """check if the TPF file exists and download it if not"""
    ID = int(ID)
    destination = path_for_epic(dire, ID, camp)
    if os.path.isfile(destination): # If the files is there, then there is nothing else to do, otherwise we need to download the data.
        return    
        
    d3 = ID % 1000
    d2 = ID % 100000
    d1 = ID-d2
    d2 -= d3
    fmt = 'http://archive.stsci.edu/missions/k2/target_pixel_files/c{0:d}/{1:d}/{2:05d}/ktwo{3:d}-c{4:d}_lpd-targ.fits.gz'
    load_url = fmt.format(camp, d1, d2, ID, camp)
    if (sys.version_info > (3, 0)):
        urllib.request.urlretrieve(load_url, destination)
    else:
        urllib.urlretrieve(load_url, destination)
