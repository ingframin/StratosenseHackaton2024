#################################################################
# Automation file for the NG5182B signal generator:
# Requires:
# - PyVISA https://pyvisa.readthedocs.io/en/latest/index.html
# - NI-VISA, Keysight VISA, R&S VISA, tekVISA or pyvisa-py
# Works on both Windows, MacOS and Linux
#################################################################

import pyvisa as pv

class SignalGenerator:

    def __init__(self,con_string, rm=None) -> None:
        if rm is None:
            self.rm = pv.ResourceManager('@py')#temporary
        else:
            self.rm = rm
        

        self.sg = self.rm.open_resource(con_string)
        self.id = self.sg.query('*IDN?')
        print(self.id)
        self.out_on_off = False

    def set_frequency(self, freq:float, unit:str)->None:
        self.sg.write(":FREQ:CW "+str(freq)+' '+unit)
        #print(self.sg.query(":FREQ:CW?"))
        
    def set_amplitude(self, ampl:float, unit:str)->None:
        self.sg.write("POW:AMPL "+str(ampl)+' '+unit)
        #print(self.sg.query(":POW:AMPL?"))

    def toggle_output(self)->None:
        self.out_on_off = not self.out_on_off
        if self.out_on_off:
            self.sg.write(":OUTP:STAT ON")
        else:
            self.sg.write(":OUTP:STAT OFF")

