# AUTHOR : JOHNN 
# DATE : 13-15 MAY 2019
# heat added : 24 July 2019
# RUN code with options specified.
# Speech Segmentation
# 
# REQUIRED : can run in a new directory with dev_spk.list, test_spk.list and 

# Imports
# import scipy.signal
from scipy import signal
import h5py
import numpy as np
import soundfile as sf
import pickle
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import argparse
import os
#import keras
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
#from prepare_dataset import prepare_list
from sklearn.preprocessing import MinMaxScaler

# image_size = 299


# Functions
def createLineIterator(P1, P2, img):
    """
    The folowing is BRESENHAM'S LINE DRAWING ALGORITHM - modified by author from source on stackoverflow
    COMMENTS BELOW are from the original source:
    
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]


    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)


    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

def save_heat_image(data, cm, fn):
    """ This function saves the pixelwise image data array - eg., heat map """   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap=cm)
    plt.savefig(fn, dpi = height) 
    plt.close()
    
def plot_RPS_heat_image(args, data, label, part_type, fn):
    # define scaler, delay, n(image size), colormap
    delay = -6
    n = args.target_size
    #cm = 'hot'
    cm = 'gray'
    
    scaler = MinMaxScaler(feature_range=(2,(args.target_size)-2))
    delaydata = np.roll(data,delay)
    datapoints = np.concatenate((data.reshape((-1,1)),delaydata.reshape((-1,1))),axis=1)
    scaler.fit(datapoints)
    scaled_datapoints = scaler.transform(datapoints)
    sdn = np.int16(scaled_datapoints)
    # create image array
    img_array = np.ones([n,n], dtype=int)
    sdn_len =len(sdn)
    
    for k,point in enumerate(sdn):
        if k < (sdn_len-1):
            # send points to LineIterator
            pt_1 = sdn[k]
            pt_2 = sdn[k+1]
            lineIt = createLineIterator(pt_1,pt_2,img_array)
            for x, y, val in lineIt:
                #print(x,y,val)
                img_array[int(x)][int(y)]+=1  
                
    savepath = os.path.join(args.target_path, part_type, label)
    if not os.path.exists(savepath):
        os.makedirs(savepath)    
    savename = os.path.join(savepath,fn)
    save_heat_image(img_array,cm,savename)#'hot',   
    #return img_array
    
    
def plot_RPS_image(args, data, label, part_type, fn):
    #print(label)
    savepath = os.path.join(args.target_path, part_type, label)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    height = args.target_size
    width = args.target_size
    
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig,[0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.plot(data,np.roll(data, -6),linewidth=0.2)
    savename = os.path.join(savepath,fn)
    plt.savefig(savename, dpi = height)
    plt.close()
        
def make_image_data(args, filt_coeffs ):
    """
    function reads details from csv, sources segment wavdata, 
    filters and converts them to class-wise folders with plot images    
    """
    # read csv
    df = pd.read_csv('lists/phone_seg_details.csv')
    for i in range(df.shape[0]):
        data, fs = sf.read(df['wav_location'][i])
        wav_seg = data[df['start'][i]:df['end'][i]]
        label = df['label'][i]
        part_type = df['part_type'][i]
        fn = label+'_'+str(i)+'_'+df['wav_location'][i].replace('/','__').replace('.WAV', '.png')
        if args.num_filt == 0:
            if args.heat == True:
                plot_RPS_heat_image(args, wav_seg, label, part_type, fn)
            else:
                plot_RPS_image(args, wav_seg, label, part_type, fn)

def create_filtered_dataset_structure(args, part_lengths):
    parts = ('train','dev', 'test')
    image_size = (args.target_size, args.target_size, 3)
    # STEP 1 : file
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    fn = 'rpsseg_data_'+str(args.num_filt)+'_filt.hdf5'
    hdf_rpsseg_store = h5py.File(os.path.join(args.target_path,fn),'a')
    rps_hdf_ref = {}
    # STEP 2 : 
    rps_hdf_ref_groups = {} # intermediate storage
    for part, part_len in zip(parts, part_lengths) :
        rps_hdf_ref[part]={}
        rps_hdf_ref_groups[part] = hdf_rpsseg_store.create_group(part)
        # STEP 3 : 
        if args.num_filt == 0:
            rps_hdf_ref[part][0] = rps_hdf_ref_groups[part].create_dataset('unf', (part_len,) + image_size, dtype=np.dtype(np.uint8))
        elif args.num_filt > 0:
            for i in range(args.num_filt):
                rps_hdf_ref[part][i] = rps_hdf_ref_groups[part].create_dataset('f'+str(i), (part_len,) + image_size, dtype=np.dtype(np.uint8))
    return(hdf_rpsseg_store, rps_hdf_ref)

def plot_RPS_heat_numpy(args, data):
    # define scaler, delay, n(image size), colormap
    delay = -6
    n = args.target_size
    #cm = 'hot'
    cm = 'gray'
    
    scaler = MinMaxScaler(feature_range=(2,(args.target_size)-2))
    delaydata = np.roll(data,delay)
    datapoints = np.concatenate((data.reshape((-1,1)),delaydata.reshape((-1,1))),axis=1)
    scaler.fit(datapoints)
    scaled_datapoints = scaler.transform(datapoints)
    sdn = np.int16(scaled_datapoints)
    # Create image array
    img_array = np.ones([n,n], dtype=int)
    sdn_len =len(sdn)    
    for k,point in enumerate(sdn):
        if k < (sdn_len-1):
            # Send points to LineIterator
            pt_1 = sdn[k]
            pt_2 = sdn[k+1]
            lineIt = createLineIterator(pt_1,pt_2,img_array)
            for x, y, val in lineIt:
                # Print(x,y,val)
                if img_array[int(x)][int(y)]<=200:
                    img_array[int(x)][int(y)]+=1  
                
    """savepath = os.path.join(args.target_path, part_type, label)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = os.path.join(savepath,fn)
    save_heat_image(img_array,cm,savename)#'hot',   """
    return img_array

def plot_RPS_numpy(args,filt_data):
    width_target = args.target_size/100.0
    fig = plt.figure(figsize=(width_target,width_target), dpi=100)
    #ax = fig.add_axes([0.,0.,1.,1.])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.plot(filt_data,np.roll(filt_data, -6), linewidth = 0.9)
    fig.canvas.draw()
    figg_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figg_data = figg_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return figg_data

def scatter_RPS_numpy(args,filt_data):
    width_target = args.target_size/100.0
    fig = plt.figure(figsize=(width_target,width_target), dpi=100)
    #ax = fig.add_axes([0.,0.,1.,1.])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.plot(filt_data,np.roll(filt_data, -6), linestyle='', marker='o', markersize=0.5 )#, linewidth = 0.9)
    fig.canvas.draw()
    figg_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figg_data = figg_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return figg_data

def hexbin_RPS_numpy(args,filt_data):
    width_target = args.target_size/100.0
    fig = plt.figure(figsize=(width_target,width_target), dpi=100)
    #ax = fig.add_axes([0.,0.,1.,1.])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.hexbin(filt_data,np.roll(filt_data, -6))#, linestyle='', marker='o', markersize=0.5 )#, linewidth = 0.9)
    fig.canvas.draw()
    figg_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figg_data = figg_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return figg_data

def encode_labels(args, label_data):
    with open(args.classfile,'r') as cf:
        phnlist = cf.readlines()
    phnlist = [ x.strip() for x in phnlist ]
    num_classes = len(phnlist)
    l_encoder = LabelEncoder()
    integer_encoded = l_encoder.fit_transform(label_data)
    Y = keras.utils.to_categorical(integer_encoded,num_classes)
    return Y

def make_numpy_hdf_data(args, filt_coeffs):
    """ 
    function loads details from csv, sources segment wavdata,
    creates hdf structure, filters wavseg if required and stores them in hdf file;
    the labels are one-hot encoded and stored as hdf
    """
    # read csv
    df = pd.read_csv('lists/phone_seg_details.csv')
    # lengths of each part
    part_lengths = (df.groupby(['part_type']).size()['train'],
                df.groupby(['part_type']).size()['dev'], 
                df.groupby(['part_type']).size()['test'])
    
    # create hdf data structure
    hdf_rpsseg_store, rp_hdf_ref = create_filtered_dataset_structure(args, part_lengths)
    
    # label and indx_pointer
    lbl_train, lbl_dev, lbl_test = [],[],[]
    part_indx = {
        'train':0,
        'dev':0,
        'test':0
    }
    train_indx, dev_indx, test_indx = 0,0,0
    for i in range(df.shape[0]):
        data, fs = sf.read(df['wav_location'][i])
        wav_seg = data[df['start'][i]:df['end'][i]]
        label = df['label'][i]
        part_type = df['part_type'][i]
        if args.num_filt == 0:
            rp_hdf_ref[part_type][0][part_indx[part_type]] = plot_RPS_numpy(args, wav_seg)
            part_indx[part_type] += 1
        elif args.num_filt> 0:
            for j in range(args.num_filt):
                #rp_hdf_ref[part_type][j][part_indx[part_type]] = get_plot_data(filtered_sig[j])
                #part_indx[part_type] += 1
                pass
        if part_type == 'train':
            lbl_train.append(label)
        elif part_type == 'dev':
            lbl_dev.append(label)
        elif part_type == 'test':
            lbl_test.append(label)
        else:
            pass
    
    #encode labels to one-hot
    lbl_trn_enc = encode_labels(args, np.asarray(lbl_train))
    lbl_dev_enc = encode_labels(args, np.asarray(lbl_dev))
    lbl_test_enc = encode_labels(args, np.asarray(lbl_test))
    # create dataset for storing labels -     
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    lbl_fn = 'label_data.hdf5'
    hdf_label_store = h5py.File(os.path.join(args.target_path, lbl_fn), 'a')
    hdf_label_store.create_dataset('train', data = lbl_trn_enc)
    hdf_label_store.create_dataset('dev', data = lbl_dev_enc)
    hdf_label_store.create_dataset('test', data = lbl_test_enc)
    # Flush and close datastores
    
    hdf_rpsseg_store.flush()
    hdf_label_store.flush()
    
    hdf_rpsseg_store.close()
    hdf_label_store.close()
            
            
def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--num_filt','-q', type=int, default=1,
                        help='the number of filters in the bank [ default : 1]') 
    parser.add_argument('--classfile', '-f', type=str, default='lists/phnlist',
                       help='the phoneme classes required in the experiment [default : lists/phnlist]')
    parser.add_argument('--target_size', '-tz', type=int, default=224,
                       help='the target image size for the prepared data [default : 224 ]')
    parser.add_argument('--target_type', '-tt', type=str, default='imagefolder',
                       help=" choose target type from either 'hdf', 'imagefolder', etcc")
    parser.add_argument('--target_path', '-tpath', type=str, default='data/imdata',
                       help=" specify target path as data/imdata or data/hdfdata [ default : data/imdata ]")
    parser.add_argument('--heat', '-ht', type=bool, default=False,
                       help=" specify heat map [ True/ False ] [ default : True]")
    args = parser.parse_args()
    return args


def main():
    print('main_function')
    args = read_args()
    with open('lists/dev_spk.list','r') as lst:
        dev_spk_list = lst.readlines()
    dev_spk_list = [ spk.strip().upper() for spk in dev_spk_list ]
    # get list of test
    with open('lists/test_spk.list','r') as lst:
        test_spk_list = lst.readlines()
    test_spk_list = [ spk.strip().upper() for spk in test_spk_list ] 
    
    prepare_list(args, dev_spk_list, test_spk_list)
    
    if args.target_type == 'imagefolder':
        make_image_data(args, filt_coeffs=0)
    elif args.target_type == 'hdf':
        make_numpy_hdf_data(args, filt_coeffs = 0)
    
if __name__ == '__main__':
    main()