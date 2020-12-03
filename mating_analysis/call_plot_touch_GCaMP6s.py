import plot_touch
from plot_touch import call_plot_touch

x_dat='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/x_events'
ylims=[-0.1,0.5]
mean_dat='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/mean_female_touching_male_ipsi'

outputfile ='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/aDN_GCaMP6s_female_touching_male'

keys1=['male_n_f','male_mean_event_f','male_SEM_event_f','male_mean_event_peak_aligned_f','male_SEM_event_peak_aligned_f','male_eventpeaks_mean_f','male_eventpeaks_SEM_f']

mean_dat2='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/mean_mated_female_touching_male_ipsi'

outputfile2 ='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/aDN_GCaMP6s_mated_female_touching_male'
keys2=['male_n_mf','male_mean_event_mf','male_SEM_event_mf','male_mean_event_peak_aligned_mf','male_SEM_event_peak_aligned_mf','male_eventpeaks_mean_mf','male_eventpeaks_SEM_mf']
mean_dat3='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/mean_male_touching_male_ipsi'

outputfile3 ='/Volumes/LaCie/Projects/aDN/imaging/aDN_touch/Results_GCaMP6/aDN_GCaMP6s_male_touching_male'

keys3=['male_n_m','male_mean_event_m','male_SEM_event_m','male_mean_event_peak_aligned_m','male_SEM_event_peak_aligned_m','male_eventpeaks_mean_m','male_eventpeaks_SEM_m']


call_plot_touch(mean_dat,x_dat,outputfile,keys1,ylims)
call_plot_touch(mean_dat2,x_dat,outputfile2,keys2,ylims)
call_plot_touch(mean_dat3,x_dat,outputfile3,keys3)
