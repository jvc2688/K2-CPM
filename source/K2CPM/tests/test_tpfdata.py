import numpy as np

from K2CPM import tpfdata


def test_check_pixel_in_tpf():
    campaign = 92
    channel = 31
    pix_y = 670#265
    pix_x = 883#494
    epic_id = 200071074#200069761
    tpf_dir = 'tpf/'
    
    tpfdata.TpfData.directory = tpf_dir

    tpf_data = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)

    assert tpf_data.check_pixel_in_tpf(column=pix_y, row=pix_x) is True
    assert tpf_data.check_pixel_in_tpf(column=pix_x, row=pix_y) is False
