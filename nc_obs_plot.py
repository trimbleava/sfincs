
"""
File name: nc_obs_plot.py
Description: Graphically, compares results of observed data versus a model output.
Author: Beheen Trimble
Date created: 3/14/2020
Python Version: 3.7
"""

import sys, os, logging, logging.config
#import re, ntpath
#import time, math
import urllib.request
from datetime import datetime, timedelta
from optparse import OptionParser

import numpy as np
#import numpy.ma as ma
import pandas as pd
from netCDF4 import Dataset, num2date, date2index 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from scipy.interpolate import UnivariateSpline

# local import
import utils
#from plot_data import plot_multi_yyymmddhm_with_stat, plot_scatter_shell

log_file = "nc_obs_plot.log"
logger = logging.getLogger(__name__)


# To run: python nc_obs_plot.py -m sfincs -n sandy -s "20121025,20121104" -f sfincs_map.nc -v zs -o outdir

def handle_command_line():

    usage = '\n%s --model_name <sfincs|dflow> (optional) ' \
                  '--hurricane_name <isabel|sandy|irene> (optional) '\
                  '--storm_time_range <"20030910,20030924"> (optional) ' \
                  '--file.nc <netcdf file name> ' \
                  '--ncvar <netcdf variable to plot> (optional) ' \
                  '--out_dir <output directory> (optional)' % sys.argv[0]

    usage += "\n"

    parser = OptionParser(usage=usage)

    parser.add_option("-m", "--model_name",
                      type="string",
                      action="store",
                      dest="model_scenario",
                      default='sfincs',
                      help="Model name [default: %default]")

    parser.add_option("-n", "--hurricane_name",
                      type="string",
                      action="store",
                      dest="hurricane_name",
                      default='sandy',
                      help="Hurricane to graph [default: %default]")


    parser.add_option("-s", "--storm_time_range",
                      type="string",
                      action="store",
                      dest="storm_date_time",
                      default="20121025,20121104",
                      help="Hurricane (storm) start and end date/time [default: %default]")

    parser.add_option("-f", "--file.nc",
                      action="store",  # optional because action defaults to "store"
                      dest="hist_file",
                      help="Provide netcdf file name to plot" )

    parser.add_option("-v", "--ncvar",
                      action="store",  # optional because action defaults to "store"
                      dest="ncvar_name",
                      default='zs',
                      help="Provide netcdf file variable to plot [default: %default]")

    parser.add_option("-o", "--out_dir",
                      action="store",
                      dest="out_dir",
                      default=os.path.abspath(os.path.dirname(__file__)),
                      help="Output directory where output graphs will reside [default: %default]")

    (opts, args) = parser.parse_args()

    # the only requirement is the ncfile, var name,  and hurricane name(needed for title)
    if not  opts.hist_file:               # or not opts.hurricane_name or not opts.ncvar_name :
        parser.error(usage)

    hurricane_name = opts.hurricane_name.lower()
    if 'isabel' in hurricane_name or \
        'sandy' in hurricane_name or \
        'irene' in hurricane_name:
        pass
    else:
        parser.error("Unknown hurricane name %s. Only 3 hurricanes is implemented: Isabel, Sandy, Irene." % hurricane_name)

    
    if opts.out_dir:
        out_dir = opts.out_dir
    else:
        # not out_dir specified by the user, output in system output
        out_dir = os.path.dirname(os.path.abspath(opts.hist_file)) 

    if os.path.isdir(out_dir):
        if os.access(out_dir, os.W_OK and os.X_OK):
            pass
    else:
        parser.error("Output directory does not exists or you do not have access write to %s." % out_dir)


    storm_time = storm_stime = storm_etime = None
    if opts.storm_date_time:
        storm_time = opts.storm_date_time
        try:
            storm_stime, storm_etime = storm_time.split(",")
            storm_stime = storm_stime.strip()
            storm_etime = storm_etime.strip()
        except:
            parser.error("Invalid storm date/time format %s\nExpected date/time format is like %s" % (storm_time, '20121025,20121104'))


    user_datetime = (storm_stime,storm_etime)
    print("User Date   :", user_datetime,"\n")


    # check input file
    global dbaync_fptr
    dbay_hist_nc = opts.hist_file

    if not os.path.isfile(dbay_hist_nc):
        parser.error("Input file %s does not exists. Can not continue." % dbay_hist_nc)
    else:
        dbaync_fptr = Dataset(dbay_hist_nc, 'r')

    # check the variable, start and end date in ncfile 
    try:
        var = dbaync_fptr.variables[opts.ncvar_name]
    except KeyError as e:
        parser.error("Variable {} is not found in netcdf file\n".format(opts.ncvar_name))
      
   
    kwarg = compare_datetime(user_datetime)
    if kwarg["compare"] == False:
        parser.error(kwarg["msg"])

    
    return (opts.model_scenario, hurricane_name, opts.ncvar_name, out_dir, dbay_hist_nc, kwarg)


"""
finds start/end time upto seconds in the ncfile
to compare with the dates given by the user.
The calling method should do the following:
if the timing do not match but there is subset 
of given time, proceed. Otherwise gives error.
Assumes the ncfile is hourly based, same as hourly 
observation data
"""
def find_ncfile_time(timevar='time'):

    nctime_var = dbaync_fptr.variables['time']
    nctime_val = nctime_var[:]
    nctime_un = dbaync_fptr.variables['time'].units

    # I think, if known from ncfile that date column is in number, like 3600, 7200, ..., then we can use num2date!
    # Test on this with other ncfiles
    nc_time = num2date(nctime_val,nctime_un)

    nc_stime = nc_time[0]
    nc_etime = nc_time[-1]
    interval = nc_time[1] - nc_stime
    sec = interval.total_seconds()
    # diff = round((sec - 3600),1) 
    start_time = end_time = 0

    # TO DO - check with other ncfile where time is not like 3600, 7200, 10800, ... second since ... 
    # start time
    #if 'seconds since' in nctime_un.lower():
    start_time = nc_stime
    nc_sindex = date2index(start_time, nctime_var)
    
    # end time
    end_time = nc_etime          
    nc_eindex = date2index(end_time, nctime_var)
     
    ncfile_datetime = (start_time.strftime('%Y%m%d'), nc_sindex, end_time.strftime('%Y%m%d'), nc_eindex)
    return (ncfile_datetime)


"""
finds if user entered date time for a give storm matches the 
model output (hist.nc) date/time. If they don't match uses the
ncfile datetime. Honors subset of date/time.
"""
def compare_datetime(user_datetime):

    # need to make sure the timing within the dataset matches the storm selected by the command line
    # 2012-10-22 07:00:00 2012-11-02 05:54:00
    
    ncfile_datetime = find_ncfile_time()
    nc_stime, nc_sindex, nc_etime, nc_eindex = ncfile_datetime
    
    if nc_stime == -1:
        msg = "Error from ncfile datetime\n"
        kwarg = {"compare": False, "msg": msg}
        return kwarg

    #print("Netcdf Date : ('{}','{}')\n".format(nc_stime, nc_etime))
    print("NetCDF Date :",(nc_stime,nc_etime),"\n")
   
    usr_stime = nc_stime; usr_sindex = nc_sindex
    usr_etime = nc_etime; usr_eindex = nc_eindex

    usr_stime_str, usr_etime_str = user_datetime
    nctime_var = dbaync_fptr.variables['time']
    
    if usr_stime_str is not None:
        user_stime = datetime.strptime(usr_stime_str,'%Y%m%d')
        try:
            usr_sindex = date2index(user_stime, nctime_var)
            usr_stime = datetime.strftime(user_stime, '%Y%m%d')
        except:
            # ValueError: Some of the times given are before the first time in `nctime`.
            print("Warning: user start time {} not found in netcdf file".format(usr_stime_str))
            print("Setting start time to {}\n".format(nc_stime))
            usr_stime = nc_stime

    if usr_etime_str is not None:
        user_etime = datetime.strptime(usr_etime_str,'%Y%m%d')
        try:
            usr_eindex = date2index(user_etime, nctime_var)
            usr_etime = datetime.strftime(user_etime, '%Y%m%d')
        except:
            print("Warning: user end time {} not found in netcdf file".format(usr_etime_str))
            print("Setting end time to {}\n".format(nc_etime))
            usr_etime = nc_etime       
    
    print("\nApplied Date: ", usr_stime, usr_sindex, usr_etime, usr_eindex)

   

    kwarg = {"compare": True}     # hold the result

    kwarg["storm_sidx"] = usr_sindex
    kwarg["storm_eidx"] = usr_eindex
    kwarg["storm_sdate"] = usr_stime
    kwarg["storm_edate"] = usr_etime

    return kwarg

    

def station_to_filename(station_name):
    """
    Pre process the station name to remove white space and commas.
    The processed name is being used as the file name.
    :param station_name: A list of pre-defined stations in Delaware River.
    :return: cleaned up name acceptable by unix file name syntax
    """
    sta_no_comma = station_name.split(",")
    space_to_underscore = sta_no_comma[0].replace(' ', '_').lower()
    return space_to_underscore



def write2file(data, major_title, title, outfile_name, outfile_ext):
    """
    Extracted data from model output netcdf file and extracted data from
    NOAA site are saved into a file per observation station
    :param data: array of date1, data1, date2, data2
    :param major_title:
    :param title:
    :param outfile_name:
    :param outfile_ext: usually .csv
    :return: None
    """
    frmt = "%s, %8.3f, %s, %8.3f"

    for key in data.keys():
        out = outfile_name
        out += key + outfile_ext
        df = data[key]
        df.to_csv(out, index = False)
        """
        optr = open(out, "w+")
        heading = major_title + key.title()
        optr.write(heading + "\n")
        optr.write(title + "\n")
        np.savetxt(optr, data[key], delimiter=",", fmt=frmt, newline="\n")
        optr.close()
        """


# IMPORTNT NOTE: TODO - currently, data for hurricane Sandy comes as nagative
# so to fix this for now we keep changing the sign in this function. See line
# dflow_data = station_data[:, 3] * -1 - remove ( * -1) when Isabel!!

def tsplot(hurricane_name, station_data, plot_file_name, plt_title, plt_subtitle, ylabel):
    """
    prepares dataset per hurricane for all the pre-defined observation stations.
    :param hurricane_name: Isabel, or Sandy, or Irene, ...
    :param stations_data: Data for all the stations are extracted to be consumed
            by this function
    :param plot_file_name: program creates unique file names for each plot
    :param plt_title: plot tylored title
    :return: None
    """
    for station in station_data.keys():

        df  = station_data[station]

        plot_name = plot_file_name + station + ".pdf" 

        print("\tPlotting station %s" % station)

        try:
            _sta = station.split("_")
            sta = " ".join([s for s in _sta])
        except:
             sta = station

        plt_subtitle += " - Station " + sta.title()

        # Turn interactive plotting off
        plt.ioff()

        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.dates as mdates

        # column names 

        cols = [df.columns[0],df.columns[1]]
        
        with PdfPages(plot_name) as pdf:
      
            fig, ax = plt.subplots(figsize=(20, 10))
            ax = df[cols[0]].plot( marker='.', markersize=0, linestyle='-', linewidth=0.5, color="C3", label='Observed')
            df[cols[1]].plot(ax=ax, marker='o', markersize=1, linestyle='-', linewidth=0.5, color="C2", label='Predicted')

            ax.legend()
            ax.set_title(plt_title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Date')
            #ax.set_ylim(0, 0.3)
            
            pdf.savefig()
            plt.xticks(rotation=0)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
            plt.close()


def process_data(station_ids, station_names, station_locs, mode, start_date_idx, end_date_idx, start_date, end_date, datum, ncvar):
    """
    Extract tide data from NOAA database for time period specified by user for a specified mode (i.e storm or spinup).
    Also, Extract model output data for the same time period and same mode.
    Note: The date frequencies on tide versus model prediction may not be the same

    :param station_ids: A list of pre-defined (by NOAA) stations ids in Delaware River
    :param station_names: A list of stations names corresponding to station ids in Delaware River
    :param mode: predications or water_level (these are pre-defined keywords from NOAA database, do not change)
    :param start_date_idx: index of start date of the record in netcdf file - need index to get data from netcdf
    :param end_date_idx: index of end date of the record in netcdf file
    :param start_date: start date of the record - need date to get data from NOAA database
    :param end_date: end date of the record - need date to get data from NOAA database
    :return: numpy array of 4 columns (date, tide, date, water_levl)
    """

    data_dict = {}  # holds all the stations data

    #t_var = dbaync_fptr.variables['time']
    #dt = num2date(t_var[start_date_idx:end_date_idx],t_var.units)
    #date_time = [ datetime.strftime(x,'%Y-%m-%d %H') for x in dt]
    #df = pd.DataFrame(date_time, columns=['Datetime'])
    
    for j, sid in enumerate(station_ids):

        station_name = station_names[j]
        sta_name = station_to_filename(station_name)
        n,m = station_locs[j]

        print("Prepare to extract data from NOAA database ..........\n")

        # Note! The default interval is 6 minute interval and there is no need to specify it.
        # The hourly interval is supported for Met data and Harmonic Predictions data only
        # url can not accept space in it (i.e. end_date + something vs. 'end_date  something'
        url = "https://tidesandcurrents.noaa.gov/api/datagetter?"
        url += "begin_date=" + start_date + "&end_date=" + end_date
        url += "&station=" + str(sid)
        url += "&product=" + mode + "&datum=" + datum + "&units=metric&time_zone=gmt&application=nwc&format=csv"
        print(url)

        
        # tide data from website - one must know the format inside
        content = urllib.request.urlopen(url).read()

        if len(content) == 0:
            print("Content not found for station %s" % station_name)
            continue

        # binary content to text
        data = content.decode('UTF-8')
        lines = data.split("\n")
        df = pd.DataFrame(lines)
        
        #df = pd.read_csv('coops.dat')
        print(df.columns)

        # drop extra columns Date Time, Water Level
        data = df.drop([' Sigma', ' O or I (for verified)', ' F', ' R', ' L', ' Quality '], axis=1)
        print(len(data))

        # extract water level data, per station from ncfile
        print("Extracting water level '{}' for station id {} located at '{},{}' grid\n".format(ncvar,sid,m,n))

        # n,m is the grid location (col,row) of the station
        #n = 1000, m = 2167
        wl = dbaync_fptr.variables[ncvar][start_date_idx:end_date_idx, int(m), int(n)]
        print(len(wl))
        col = ncvar.upper()
        data.insert(2, column=col, value=wl)
        print(data.head(5))
        print(data.columns)
        
        data = data.astype({'Date Time': np.datetime64, str(col):np.float64, ' Water Level': np.float64})   # space in Wate Level is needed
        data = data.set_index('Date Time')
        
        # save this station with its data globally
        data_dict[sta_name] = data

    return data_dict



def main():

    logging.basicConfig(level="DEBUG", filename=log_file, filemode="a")

    model, hurricane_name, ncvar, out_dir, dbay_hist_nc, kwarg = handle_command_line()

    start_storm_idx = kwarg["storm_sidx"]
    end_storm_idx   = kwarg["storm_eidx"]
    start_storm     = kwarg["storm_sdate"]
    end_storm       = kwarg["storm_edate"]

    # if space, substitutes with "_" for the file name and capitalize as is for the
    # title
    ms = model
    if ' ' in model:
        ms = model.replace(" ", "_")

    plt_title = plt_subtitle = None
    

    # Note: we can not use ncfile station names to extract data from NOAA due to station_ids requirement for extraction.
    # stations with same attributes in NOAA database - these stations all have same datum (NVAD)
    station_ids = (8536110, 8557380, 8537121, 8551910, 8555889)
    station_names = ["Cape May, NJ", "Lewes, DE", "Ship John Shoal, NJ","Reedy Point, DE","Brandywine Shoal Light, DE"]
    station_locs = [(717,464), (572,258), (340,838), (160,1120), (578,485)]

    data_dict = {}  # holds processed data for all modes, if we need to plot them in one figure
    outfile_dict = {}


    major_title = model + " - Hurricane " + hurricane_name.capitalize() + " - Station "
    outfile_ext = ".csv"

    modes = ["water_level"]  # these are pre-defined keyword set in NOAA database. Do not change
    for mode in modes:
        print("\nProcessing data for %s ...............\n" % mode)

        if mode == "water_level":

            title = "Date_Time\t\t\t\tStorm\t\tDateTime\t\t\t" + model
            outfile_name = os.path.join(out_dir, ms.lower() + "_" + hurricane_name + "_storm_")
            outfile_dict[mode] = outfile_name
            plt_title = "Water level (m) prediction comparison with NOAA observed data during hurricane %s\n" % hurricane_name.capitalize()
            plt_subtitle = model
            stations_data = process_data(station_ids, station_names, station_locs, mode, start_storm_idx, end_storm_idx, start_storm, end_storm, "navd", ncvar)
                                          
            data_dict[mode] = stations_data

        #check for stations_data if empty, skip from here on!
        if len(stations_data):

            # write to file
            # write2file(stations_data, major_title, title, outfile_name, outfile_ext)
   
            # plot
            print("\tPlotting stations_data to file ...............\n")
            tsplot(hurricane_name, stations_data, outfile_name, plt_title, plt_subtitle, "Water Level (m)")

    sys.exit(0)

    # different set of stations with same attributes in NOAA database - these stations all have same datum (MSL)
    # station_ids = (8555889, 8537121, 8540433, 8539094, 8548989)
    # station_names = ["Brandywine Shoal Light, DE", "Ship John Shoal, NJ", "Marcus Hook, PA",
    #                "Burlington, Delaware River, NJ", "Newbold, PA"]
    station_ids = (8539094,)
    station_names = ["Burlington, Delaware River, NJ",]

    data_dict = {}  # holds processed data for all modes, if we need to plot them in one figure
    outfile_dict = {}

    # at most expected space delimited "modelname type scenario" or what ever name
    # if space, substitutes with "_" for the file name and capitalize as is for the
    # title
    ms = model_scenario
    if ' ' in model_scenario:
        ms = model_scenario.replace("_", " ")

    major_title = model_scenario + " - Hurricane " + hurricane_name.capitalize() + " - Station "
    outfile_ext = ".csv"

    modes = ["predictions", "water_level"]  # these are pre-defined keyword set in NOAA database. Do not change

    for mode in modes:
        print("\nProcessing data for %s ...............\n" % mode)
        if mode == "predictions":
            if start_spin_idx == -2 or end_spin_idx == -2:
                continue
            title = "Date_Time\t\t\t\tSpinup\t\tDateTime\t\t\tD-Flow"
            outfile_name = os.path.join(out_dir, ms.lower() + "_" + hurricane_name + "_spinup_")
            outfile_dict[mode] = outfile_name
            plt_title = "Tide prediction comparison with NOAA observed data during spin-up for hurricane %s\n" % hurricane_name.capitalize()
            plt_subtitle = model_scenario

            # process data from tide and from model water level, return numpy array for further processing
            # By default extracts tide data in csv format. To customize pass the outfile_ext.
            stations_data = process_data(station_ids, station_names, mode, start_spin_idx, end_spin_idx, start_spin,
                                         end_spin, "msl")
            data_dict[mode] = stations_data

        elif mode == "water_level":
            if start_storm_idx == -2 or end_storm_idx == -2:
                continue

            title = "Date_Time\t\t\t\tStorm\t\tDateTime\t\t\tD-Flow"
            outfile_name = os.path.join(out_dir, ms.lower() + "_" + hurricane_name + "_storm_")
            outfile_dict[mode] = outfile_name
            plt_title = "Water level (m) prediction comparison with NOAA observed data during hurricane %s\n" % hurricane_name.capitalize()
            plt_subtitle = model_scenario
            stations_data = process_data(station_ids, station_names, station_locs, mode, start_storm_idx, end_storm_idx, start_storm,
                                         end_storm, "msl", ncvar)
            data_dict[mode] = stations_data

            # check for stations_data if empty, skip from here on!
        if len(stations_data):
            # write data to file -
            print("\tWriting stations_data to file ...............\n")
            #write_data_tofile(stations_data, major_title, title, outfile_name, outfile_ext)

            # plot
            print("\tPlotting stations_data to file ...............\n")
            #plot_dflow(hurricane_name, stations_data, outfile_name, plt_title, plt_subtitle)



# To run: python nc_obs_plot.py -n isabel -s "20030911,20030924" -f DFlow_his.nc -v zs -o /media/sf_PROJECTS/dflow_data/output
if __name__ == "__main__":
    main()
    dbaync_fptr.close()
