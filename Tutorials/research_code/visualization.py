import  matplotlib.pyplot as plt

def showConfustionMatrix(c_mat, fig_x = 8, fig_y = 8, save_path = ""):
    '''
    Show (and save)  
        Parameters:
        cmat: Numpy 2D matrix (Num data points, Num data points), dtype = int64  
            (True label, prediction) confusion matrix
        fig_x: float (Default = 8)
            Width of the figure in inch 
        fig_y: float (Default = 8)
            Height of the figure in inch 
        save_path: string (Default = "")
            name of the saved image file if not equal to "" 
        
    '''
    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.subplot(111)
    cax = ax.imshow(c_mat, interpolation = 'none',origin = 'lower', vmin = 0, vmax = c_mat.max())
    ax.set_xlabel('Actual Class',fontsize = 16)
    ax.set_ylabel('Predicted Class',fontsize = 16)
    plt.colorbar(cax)
    if len(save_path) > 0:
        plt.savefig(save_path)
    plt.show()

