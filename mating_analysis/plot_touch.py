import scipy
import matplotlib.pyplot as plt    
from scipy import io
from functools import wraps
from datetime import datetime




def logDecorator(logfile='plot_touch_logfile.log'):
    def logwriter(a_func):
        @wraps(a_func)
        def wrapWithLog(*args,**kwargs):
            now = str(datetime.now())
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_string = a_func.__name__ + " was called on the following data:"
            print(dt_string)
            print(log_string)
            for num in args:
                print(num)
         
        
            # Open the logfile and append
            with open(logfile, 'a') as opened_file:
                # Now we log to the specified logfile
                opened_file.write(dt_string + '\n')
                opened_file.write(log_string + '\n')
                for num in args:
                    opened_file.write(str(num)+'\n' )
        return wrapWithLog
    return logwriter



def plot_touch(mean_data,x_data,outfile,keys,ylims=[-0.1,0.5]):
    
    x= [data[0] for data in x_data['xevents_nonempty'][0][0]]
    n= mean_data[keys[0]][0][0]
    mean_touch_aligned=[data[0] for data in mean_data[keys[1]]]
    SEM_touch_aligned= [data[0] for data in mean_data[keys[2]]]
    
    plus_SEM_touch_aligned=[m+S for m,S in zip(mean_touch_aligned,SEM_touch_aligned)]
    minus_SEM_touch_aligned=[m-S for m,S in zip(mean_touch_aligned,SEM_touch_aligned)]
    data_touch_aligned = [mean_touch_aligned,plus_SEM_touch_aligned,minus_SEM_touch_aligned]
    name_touch_aligned= '_touch_aligned.eps'


    mean_peak_aligned=[data[0] for data in mean_data[keys[3]]]
    SEM_peak_aligned= [data[0] for data in mean_data[keys[4]]]
    
    plus_SEM_peak_aligned=[m+S for m,S in zip(mean_peak_aligned,SEM_peak_aligned)]
    minus_SEM_peak_aligned=[m-S for m,S in zip(mean_peak_aligned,SEM_peak_aligned)]
    data_peak_aligned = [mean_peak_aligned,plus_SEM_peak_aligned,minus_SEM_peak_aligned]
    name_peak_aligned= '_peak_aligned.eps'

    datalist = [data_touch_aligned,data_peak_aligned]
    namelist = [name_touch_aligned,name_peak_aligned]

    mean_peak=mean_data[keys[5]][0][0]
    SEM_peak= mean_data[keys[6]][0][0]
    
    for data,name in zip(datalist,namelist):
        mean = data[0]
        plus_SEM = data[1]
        minus_SEM=data[2]
        outfilename=outfile+name



    
        ax = plt.subplot(111)
        ax.tick_params(axis='both', direction='out')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(ylims)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(2)
    
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        ax=plt.fill_between(x,minus_SEM,plus_SEM,color='lightgrey')
        ax=plt.plot(x,mean,color='black')
        plt.text(-2,-0.06,("n = "+str(n)))


        plt.savefig(outfilename)
        plt.close()
    return

@logDecorator()
def call_plot_touch(datapath,xpath,outfilepath,keys,ylims):
    means=scipy.io.loadmat(datapath,matlab_compatible=True)
    x=scipy.io.loadmat(xpath,matlab_compatible=True)
    
    plot_touch(means,x,outputfilepath,keys,ylims)


