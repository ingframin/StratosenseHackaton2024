from .data_model import Gval

def load_Gvals(filename:str)->list[Gval]:
    f = open(filename)
    lines = f.readlines()
    f.close()
    #the replace is in case the Gvals are just printed on file rather than serialised
    lines = [l.replace("'",'"') for l in lines]
    gv = []
    for l in lines:
        gv.append(Gval.from_json(l))
    return gv

def get_min_dif(Ptrg:float, pows:list[float])->int:
    min_p:float = abs(Ptrg-pows[0])
    ind:int = 0
    rind:int = 0
    for p in pows:
        dif = abs(Ptrg-p)
        if dif < min_p:
            min_p = dif
            rind = ind
        ind+=1
    return rind

def find_most_common(iterable_list): 
    most_common_element = max(set(iterable_list), key = iterable_list.count) # Returns the most common element
    return most_common_element # Return most common element

def load_fg(filename:str)->dict[int,float]:
    fg = {}
    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            ls = line.strip().split()

            fg[int(ls[0])] = float(ls[1])

    return fg

def find_gain(fg:dict[int,float], freq:int)->float:
    if freq in fg:
        return fg[freq]
    f1 = 0
    f2 = 0
    for f in fg:
        if f<freq:
            f1 = f
        if f>freq:
            f2 = f
            break
    g1 = fg[f1]
    g2 = fg[f2]
    return (g2+g1)/2

def find_best_gain(Ptrg:float, g_vals:list, frequency:list)->Gval:
    pows = []
    gains = []
    for gv in g_vals:
        if gv.frequency == frequency and gv.P_target == Ptrg:
            pows.append(gv.P_dBm)
            gains.append(gv.gain)

    ind = get_min_dif(Ptrg,pows)
    return gains[ind]

class FindGain:
    def __init__(self,fg:dict[float,float]) -> None:
        self.fg:dict[float,float] = fg

    # Apply memoization to speed up computation
    def __call__(self, frequency:float)->float:
        if frequency not in self.fg:
            f1 = 0.0
            f2 = 0.0
            for f in self.fg:
                if f<frequency:
                    f1 = f
                if f>frequency:
                    f2 = f
                    break

            g1 = self.fg[f1]
            g2 = self.fg[f2]
            self.fg[frequency] = (g2+g1)/2

        return self.fg[frequency]

    