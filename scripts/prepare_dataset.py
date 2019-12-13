__author__ = 'joerg+john'

import os, csv
import numpy as np
import argparse
    
rootDir = '/data/TIMIT_standard/'

def get_estimate_size(args, dev_spk_list, test_spk_list):
    train_seg_count = 0
    dev_seg_count = 0
    test_seg_count = 0
    with open(args.classfile,'r') as cf:
        phnlist = cf.readlines()
    phnlist = [ x.strip() for x in phnlist ]
    num_classes = len(phnlist)
    dirlist = [ 'DR1', 'DR2', 'DR3', 'DR5', 'DR4', 'DR6', 'DR7', 'DR8' ]
    dirlis = ['DR8']
    for d in dirlist:
        for dirName, subdirList, fileList in os.walk( rootDir + "TRAIN/" + d + "/"):
            print("Found directory %s"% dirName)
            path,folder_name = os.path.split(dirName)
            #print("Speaker: " + folder_name)        
            if folder_name.__len__() >=1:
                temp_name = ""
                for fname in sorted(fileList):
                    name = fname.split(".")[0]
                    if name != temp_name:
                        # don't take in SA sentences
                        if not 'SA' in name:
                            temp_name = name
                            #print('\t%s' % dirName+"/"+name)
                            #wav_location = dirName+"/"+name+".WAV"
                            phn_location = dirName+"/"+name+".PHN"
                            with open(phn_location,'r') as phn:
                                phns = phn.readlines()                            
                            phns = [ x.strip().split(' ')[2] for x in phns ]
                            for phn_ele in phnlist:
                                train_seg_count += phns.count(phn_ele)
                            #train_seg_count += (len(phns) - phns.count('q'))
    for d in dirlist:
        for dirName, subdirList, fileList in os.walk( rootDir + "TEST/" + d + "/"):
            print("Found directory %s"% dirName)
            path,folder_name = os.path.split(dirName)
            #print("Speaker: " + folder_name)        
            if folder_name.__len__() >=1:
                temp_name = ""
                for fname in sorted(fileList):
                    name = fname.split(".")[0]
                    if name != temp_name:
                        # don't take in SA sentences
                        if not 'SA' in name:
                            temp_name = name
                            #print('\t%s' % dirName+"/"+name)
                            #wav_location = dirName+"/"+name+".WAV"
                            phn_location = dirName+"/"+name+".PHN"
                            with open(phn_location,'r') as phn:
                                phns = phn.readlines()
                            phns = [ x.strip().split(' ')[2] for x in phns ]
                            if folder_name in dev_spk_list:
                                for phn_ele in phnlist:
                                    dev_seg_count += phns.count(phn_ele)
                                #dev_seg_count += (len(phns) - phns.count('q'))
                            elif folder_name in test_spk_list:
                                for phn_ele in phnlist:
                                    test_seg_count += phns.count(phn_ele)
                                #test_seg_count += (len(phns) - phns.count('q'))
    return(train_seg_count, dev_seg_count, test_seg_count)

def prepare_list(args, dev_spk_list, test_spk_list):
    fcsv = open('lists/phone_seg_details.csv','w')
    fcsv.write('sent_id,label,start,end,wav_location,part_type\n')
    train_seg_count = 0
    dev_seg_count = 0
    test_seg_count = 0
    with open(args.classfile,'r') as cf:
        phnlist = cf.readlines()
    phnlist = [ x.strip() for x in phnlist ]
    num_classes = len(phnlist)
    dirlist = [ 'DR1', 'DR2', 'DR3', 'DR5', 'DR4', 'DR6', 'DR7', 'DR8' ]
    dirlis = ['DR8']
    for d in dirlist:
        for dirName, subdirList, fileList in os.walk( rootDir + "TRAIN/" + d + "/"):
            print("Found directory %s"% dirName)
            path,folder_name = os.path.split(dirName)
            #print("Speaker: " + folder_name)        
            if folder_name.__len__() >=1:
                temp_name = ""
                for fname in sorted(fileList):
                    name = fname.split(".")[0]
                    if name != temp_name:
                        # don't take in SA sentences
                        if not 'SA' in name:
                            temp_name = name
                            #print('\t%s' % dirName+"/"+name)
                            wav_location = dirName+"/"+name+".WAV"
                            phn_location = dirName+"/"+name+".PHN"
                            sent_id = dirName+"/"+name
                            with open(phn_location,'r') as phn:
                                phns = phn.readlines()
                                
                            part_type = 'train'
                            for ele in phns:
                                start, end, label = ele.strip().split(' ')
                                if label in phnlist:                                    
                                    #sent_id = dirName+"/"+name
                                    fcsv.write(sent_id+','+label+','+start+','+end+','+wav_location+','+part_type+'\n')
    for d in dirlist:
        for dirName, subdirList, fileList in os.walk( rootDir + "TEST/" + d + "/"):
            print("Found directory %s"% dirName)
            path,folder_name = os.path.split(dirName)
            #print("Speaker: " + folder_name)        
            if folder_name.__len__() >=1:
                temp_name = ""
                for fname in sorted(fileList):
                    name = fname.split(".")[0]
                    if name != temp_name:
                        # don't take in SA sentences
                        if not 'SA' in name:
                            temp_name = name
                            #print('\t%s' % dirName+"/"+name)
                            wav_location = dirName+"/"+name+".WAV"
                            phn_location = dirName+"/"+name+".PHN"
                            sent_id = dirName+"/"+name
                            with open(phn_location,'r') as phn:
                                phns = phn.readlines()
                                
                            if folder_name in dev_spk_list:
                                part_type = 'dev'
                                for ele in phns:
                                    start, end, label = ele.strip().split(' ')
                                    #if not label == 'q' and label in phnlist: if q not required remove from phnlist
                                    if label in phnlist:                                        
                                        fcsv.write(sent_id+','+label+','+start+','+end+','+wav_location+','+part_type+'\n')                           
                            elif folder_name in test_spk_list:
                                part_type = 'test'
                                for ele in phns:
                                    start, end, label = ele.strip().split(' ')
                                    if label in phnlist:                                        
                                        fcsv.write(sent_id+','+label+','+start+','+end+','+wav_location+','+part_type+'\n')                          
    fcsv.close()
    return(train_seg_count, dev_seg_count, test_seg_count)

# Functions

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
    
if __name__ == '__main__':
    main()
