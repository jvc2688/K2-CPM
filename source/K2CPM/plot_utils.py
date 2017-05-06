import numpy as np
import matplotlib.pyplot as plt


def plot_matrix_subplots(figure, time, matrix, same_y_axis=True):
    """
    Plot given 3D matrix in subpanels. Note that 3rd component of matrix.shape 
    must be the same as time.size i.e., matrix.shape[2]==time.size
    
    TO_BE_DONE: add options to specify symbols, add errorbars, add mask options
    
    example usage:
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(50,30)
    plot_matrix(fig, time, matrix)
    plt.savefig("file_name.png")
    
    to construct input data use e.g.:
    tpfdata.TpfData.directory = TPF_DIRECTORY_NAME
    tpf = tpfdata.TpfData(epic_id=SOME_epic_id, campaign=SOME_campaign)
    time = tpf.jd_short
    matrix = tpf.get_fluxes_for_square(pix_y, pix_x, half_size=3) # 3 gives 7x7 sublots
    """
    y_lim = [np.min(matrix), np.max(matrix)]
    (i_max, j_max, _) = matrix.shape
    panels = np.flipud(np.arange(i_max*j_max).reshape(i_max, j_max)) + 1

    for i in range(i_max):
        for j in range(j_max):
            ax = plt.subplot(i_max, j_max, panels[i, j])
            ax.plot(time, matrix[i][j], '.k')
            if i != 0:
                ax.get_xaxis().set_visible(False)
            if j != 0:
                ax.get_yaxis().set_visible(False)
            if same_y_axis:
                ax.set_ylim(y_lim)
    figure.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
