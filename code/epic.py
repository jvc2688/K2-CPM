import requests
import json
import urllib
import os

def get_tpfs(ra,dec,r,camp, channel):
    url = 'https://archive.stsci.edu/k2/data_search/search.php'
    data = {"ra": ra, "dec": dec, "radius":r, "sci_campaign":camp, 'sci_channel':channel, "outputformat": "JSON", "action": "Search" }
    r = requests.post(url, data=data)
    result= json.loads(r.text)
    r_list = [int(d['K2 ID']) for d in result]
    return list(set(r_list))

def load_tpf(ID, camp, dire):
    ID = int(ID)
    d3 = ID % 1000
    d2 = ID % 100000
    d1 = ID-d2
    d2 -= d3
    file_name = 'ktwo%d-c%d_lpd-targ.fits.gz'%(ID, camp)
    if not os.path.isfile(dire+'/'+file_name):
        load_url = 'http://archive.stsci.edu/missions/k2/target_pixel_files/c%d/%d/%05d/ktwo%d-c%d_lpd-targ.fits.gz'%(camp,d1,d2,ID,camp)
        tpf_file = urllib.URLopener()
        tpf_file.retrieve(load_url, dire+'/'+file_name)