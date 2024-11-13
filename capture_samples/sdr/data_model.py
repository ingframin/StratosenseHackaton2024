from dataclasses import dataclass
import json


@dataclass
class Nval:
    frequency: float #MHz
    gain: float #dB
    P_target: float #dBm
    num_samples: int = 65536
    precision: float = 0.1 #target precision in dBm
    N: float = 1.0 #correction factor

    def __repr__(self):
        return str(self.__dict__)

    def serialize(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, js_data):
        data  = json.loads(js_data)
        return cls(data['frequency'],data['gain'],data['P_target'],data['num_samples'],data['precision'], data['N'])


@dataclass
class Gval:
    frequency: float #MHz
    gain: float #dB
    P_target: float #dBm
    P_dBm: float

    def __repr__(self):
        return str(self.__dict__)
    
    def serialize(self):
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, js_data):
        data  = json.loads(js_data)
        return cls(data['frequency'],data['gain'],data['P_target'],data['P_dBm'])
