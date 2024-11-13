import argparse
from datetime import datetime, timedelta
import json
import re

default_config = {'buffer_size': 1<<16, 'calibration_file': 'f-g_rtlsdr.txt', 'device': 'rtl', 'device_string': '', 'ext_board_port': '/dev/ttyACM0', 'measure_time': '1d', 'samplerate': 2.4, 'save_file': 'result_all_cellular_bands_2-18_rtl.txt'}

class SA:
    def __init__(self,config_file='sensor.cfg') -> None:
        parser = argparse.ArgumentParser(description='Runs a specified sdr as a electrosmog sensor.')
        parser.add_argument('--config', metavar='c', help='Configuration filename')
        parser.add_argument('--d', metavar='d', help='device must be: rtl, pluto or e312')
        parser.add_argument('--t',metavar='t', help='Time to measure: \n examples:  \n\t"45s" = 30 seconds, \n\t"30m" = 30 minutes, \n\t"6h" = 6 hours, \n\t"2d" = 2 days , \n\t"1w" = 1 week')
        parser.add_argument('--b', metavar='b', help='Buffer size,  2**b ; default = 2**12')
        parser.add_argument('--bw', metavar='bw', help='Bandwidth/samplerate in MHz; default is 2.4')
        parser.add_argument('--save_file', metavar='sf', help='Save file name')
        parser.add_argument('--cal_file', metavar='cf', help='Calibrationfile')
        parser.add_argument('--ext_board', metavar='p', help='Port of the extentionboard /dev/XXXXX ; default = /dev/ttyACM0')
        parser.add_argument('--device_string', metavar='ds', help='If needed the location of the device; for example ip:pluto.locale')
        args = parser.parse_args()
        self.config = {}
        self.load_config(args,config_file)
        

    def process_arguments(self,args):
        if args.d != None:
            self.config["device"] = args.d
        if args.t != None:
            self.config["measure_time"] = args.t
        if args.b != None:
            self.config["buffer_size"] = 1<<int(args.b)
        if args.bw != None:
            self.config["samplerate"] = float(args.bw)
        if args.save_file != None:
            self.config["save_file"] = args.save_file
        if args.cal_file != None:
            self.config["calibration_file"] = args.cal_file
        if args.ext_board != None:
            self.config["ext_board_port"] = args.ext_board
        if args.device_string != None:
            self.config['device_string'] = args.device_string
        elif self.config["device"] == 'pluto':
            self.config['device_string'] = 'ip:pluto.local'
   
        
    def load_config(self,args,config_file):

        if args.config != None:
            config_file = args.config
        # first try opening the file, if there is an error use default values and save them
        try:
            with open(config_file) as con_f:
                    self.config = json.load(con_f)
        except:
            print('Error with file. Using default settings and saving it')
            self.config = default_config
                
        self.process_arguments(args)         
        self.save_config(config_file)
        

    def save_config(self, config_file):
        with open(config_file,'w') as c_f:
            json.dump(self.config,c_f,sort_keys=True, indent=4)


    def get_time(self):
        pt = {'s': 1, 'm':60,'h':60*60,"d":60*60*24,"w":60*60*24*7}   
        res = re.findall('(\d+|[A-Za-z]+)', self.config["measure_time"])
        m_time = 0
        try: 
            for i in range(0,len(res),2):
                m_time += int(res[i])*pt.get(res[i+1])
        except:
            print("Invalid time format, terminating")
            exit()
        print("Measurement ready at:", datetime.now() + timedelta(seconds=m_time))   
        return m_time



    def get_sdr(self):
        if self.config["device"]=='rtl':
            try: # Try rtl stops when no valid sdr is detected 
                from sdr import rtl
                radio = rtl.RTL_SDR(self.get_buf_size(),ext_board_port=self.config["ext_board_port"])
            except:
                print('No RTL SDR detected')
                exit()
        elif self.config["device"] == 'pluto':
            try:
                from sdr import pluto
                radio = pluto.Pluto_SDR(self.config["device_string"],self.get_buf_size())
            except:
                print('No Pluto detected')
                exit()
        elif self.config["device"] == 'e312':
            try:
                from sdr import e312
                radio = e312.ETTUS_E312(self.get_buf_size())
            except:
                print('No e312 detected')
                exit()
        return radio


    def get_configuration(self):
        return self.get_sdr() ,self.get_config('samplerate'), self.get_buf_size() ,self.get_config('calibration_file'), self.get_time(), self.get_config('save_file')

    def get_buf_size(self):
        return self.config["buffer_size"]
    
    def get_config(self, key):
        return self.config[key]
            



    
    


    

    