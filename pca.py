#!/usr/bin/python 

from __future__ import print_function
import sys
import argparse
import csv
import numpy as np
import baseline
import matplotlib as mpl
from matplotlib import pyplot as plt

def extract_data(files,npts,skip_first,skip_last):

  # record x values, optionally dropping points from tail ends of spectrum
  # check file formatting

  with open(files[0],'r') as f:
    csv_obj = csv.reader(f,delimiter=',')
    xvals = [row[0] for row in csv_obj]

  if len(xvals) % npts == 0:
    header = 0
  elif len(xvals) % npts == 1:
    header = 1
  else:
    print('Error: csv file does not contain an appropriate number of rows')
    sys.exit()

  xvals = xvals[header+skip_first:header+npts-skip_last]

  # extract data (y values) from file/s and process into replicates

  all_data = []

  for csvfile in files: 
    try:
      with open(csvfile,'r') as f:
        csv_obj = csv.reader(f, delimiter=',')
        if header == 1: next(csv_obj)
        csv_dat = [row[1:] for row in csv_obj]
    except:
      print('Error: could not parse csv file named ' + csvfile)
      sys.exit()

  # data may be presented in an array or as concatenated vertical lists

    if len(csv_dat) == npts and len(csv_dat[0]) > 1:
      data = []
      for i in range(skip_first,npts-skip_last):
        data.append(list(map(float,csv_dat[i]))) 
      all_data.append(np.transpose(np.array(data))) 

    elif len(csv_dat) % npts == 0 and len(csv_dat) / npts > 1:
      n_rep = len(csv_dat) / npts
      data = [] 
      for i in range(0,n_rep):
        vec = [float(val[0]) for val in csv_dat[npts*i+skip_first:npts*(i+1)-skip_last]] 
        data.append(vec) 
      all_data.append(np.array(data))
  
    else:
      print('Sorry, you do not have enough data for PCA analysis')
      sys.exit()

  return all_data,np.array(list(map(float,xvals)))
      

def analyse(data):

  n = data.shape[0]
  u,s,v = np.linalg.svd(data)
  t = u*np.sqrt(n)                                       # Standardised scores
  w = np.sign(v[0,0])*np.diag(s).dot(v[0:n])/np.sqrt(n)  # Loadings

  return w,t


def make_plot(data,xlab,ylab,headings,ncolumns,filename):

  plots = plt.plot(data[:,0],data[:,1:])
  plt.xlabel(xlab,labelpad=15)
  plt.ylabel(ylab,labelpad=10)
  ax = plt.gca()
  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
  ax.legend(plots,headings[1:],prop={'size':12},ncol=ncolumns,loc="upper center")
  ylims = ax.get_ylim()
  ax.set_ylim((ylims[0],ylims[1]+200))
  plt.tight_layout()
  plt.savefig(filename,format='eps')
  plt.clf()

#---------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

  # Parse inputs

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_points', '-npt', type=int, required=True, help="Must be specified")
  parser.add_argument('--sample_files', '-samp', type=str, nargs='+', required=True, help="Must be specified")
  parser.add_argument('--reference_files', '-ref', type=str, nargs='+', default=[], help="Optional, set to zero if not specified")
  parser.add_argument('--skip_first', '-skf', type=int, default=1, help="Optional, set to 1 if not specified")
  parser.add_argument('--skip_last', '-skl', type=int, default=1, help="Optional, set to 1 if not specified")
  parser.add_argument('--name','-name', type=str, default='plot', help="Optional, name for labelling plots")
  parser.add_argument('--num_components', '-npc', type=int, default=1, help="Optional, set to 1 if not specified")
  parser.add_argument('--lambda_val', '-lam', type=float, default=1e6, help="Optional, set to 10^6 if not specified")
  args=parser.parse_args() 
  
  # Process inputs

  zero_ref = False
  if len(args.reference_files) == len(args.sample_files):
    file_set = [[ref,samp] for (ref,samp) in zip(args.reference_files,args.sample_files)]
  elif len(args.reference_files) == 1:
    file_set = [args.reference_files + args.sample_files]
  elif len(args.reference_files) == 0:
    zero_ref = True
    file_set = [args.sample_files]
  else:
    print('Error: could not unambiguously match reference files and sample files')
    print('You must have either the same number of each, a single reference or no reference')
    sys.exit()

  # Process and accumulate data

  baseline_corrected_spectra = []
  variance_loadings = []
  spectra_headings = ['']
  variance_headings = ['']

  for files in file_set:

    # For each set of files, load data sets as list of n(replicates) x p(wavenumbers) matrices
    # The first set of data is special, it's the reference set (zero-ref case inserted later)

    data,xvals = extract_data(files,args.num_points,args.skip_first,args.skip_last)

    # Construct labels for processed spectra and principal component variances

    labels = [filename.split('.')[0] for filename in files]
    if zero_ref: labels = ['Zero'] + labels
    spectra_headings += labels
    variance_headings += [label+'-'+labels[0] for label in labels[1:]] 

    # Do baseline correction for each sample set independently
    # Accumulate baseline corrected average spectra on the way through

    bl_data = []
    for data_set in data:
      pca_loadings,pca_scores = analyse(data_set)
      yvals = pca_loadings[0]
      bl = baseline.derpsalsa(xvals,yvals,als_lambda=args.lambda_val)
      data_set -= bl.T       
      baseline_corrected_spectra.append(yvals-bl)
      bl_data.append(data_set)

    # Special case insertion of zero reference spectrum + data

    if zero_ref:
      baseline_corrected_spectra = [np.zeros_like(baseline_corrected_spectra[0])] + baseline_corrected_spectra
      bl_data = [np.zeros_like(bl_data[0])] + bl_data

    # Concatenate reference and sample data, subtract reference spectrum, analyse variance

    for data_set in bl_data[1:]:
      combined_data = np.concatenate([bl_data[0],data_set]) - baseline_corrected_spectra[0]
      combined_pca_loadings,combined_pca_scores = analyse(combined_data)
      variance_loadings.append(combined_pca_loadings)

  # Print out and plot baselined "primary PCA loading" spectra

  bl_spec = np.array([xvals]+baseline_corrected_spectra).T
  np.savetxt(args.name+'_spectra.txt',bl_spec,header=','.join(spectra_headings))
  make_plot(bl_spec,'Raman Shift (cm$^{-1}$)','Intensity (arb.)',spectra_headings,len(file_set),args.name+'_spectra.eps')

  # Print out and plot variances between reference and sample data sets

  for i in range(0,args.num_components):
    variances = [xvals]
    for var_load in variance_loadings:
      variances.append(var_load[i])
    var_T = np.array(variances).T
    np.savetxt(args.name+'_variances_'+str(i)+'.txt',var_T,header=','.join(variance_headings))
    make_plot(var_T,'Raman Shift (cm$^{-1}$)','Intensity (arb.)',variance_headings,len(file_set),args.name+'_variances_'+str(i)+'.eps')

