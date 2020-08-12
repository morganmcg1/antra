'''
    Functions do process translations made by the model, removing special chars etc
'''
import re

def fastai_process_trans(trans):
    trans_ls=[]
    for s in trans: 
        #print(s)
        tmp = s.replace('xxbos','')
        tmp = tmp.replace('xxeos','')
        tmp = tmp.replace(' .','.')
        tmp = tmp.replace(' ,',',')
        tmp = tmp.replace(' ?','?')
        tmp = tmp.replace(' !','!')
        #print(tmp[0])
        if tmp.endswith('. '): tmp=tmp[:-1]
        if tmp.endswith('? '): tmp=tmp[:-1]
        if tmp.endswith('! '): tmp=tmp[:-1]

        for spec in ['xxmaj ', 'xxup ']:
            found=[]
            for m in re.finditer(spec, tmp):
                found.append(m.start())
            for f in found:
                m = tmp.find(spec)
                if m != -1:   
                    ml = m+len(spec)
                    tmp = tmp[:ml] + tmp[ml].upper() + tmp[ml+1:]
                    if m != 0:
                        tmp = tmp[:m] + tmp[ml:]
                    else: 
                        tmp = tmp[ml:]

        found=[]    
        xxwrep = 'xxwrep '            
        for m in re.finditer(xxwrep, tmp):
            found.append(m.start())
        for f in found:
            m = tmp.find(xxwrep)
            n = int(tmp[m+7])    # number of repetitions of word
            pwrep = m+8    # position where repeated word starts
            wrep = tmp[pwrep:].split()[0]    # word to be repeated
            lwrep = len(wrep)    # length of repeated word
            tmp = tmp[:m] + f"{wrep} " * n + tmp[pwrep+lwrep+1:]
        
        # Remove space at start
        if tmp[0] == ' ': tmp = tmp[1:]            
        trans_ls.append(tmp)
    return trans_ls