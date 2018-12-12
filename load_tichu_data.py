import numpy as np
import json


def parse_sql_line(sqlline):
    """
    Input: The data insert line for the .sql script as string
    Output:
        list of json format, each one is one match.
    """
    def into_json(datastr):
        """
        Given a string of raw match data, parse and input into json format
        """
        datastr = datastr.replace('\\','')
        temp = datastr.replace('\'','"')
        si = next(i for i,s in enumerate(datastr) if s=='{')
        ei = [i for i,s in enumerate(datastr) if s=='}'][-1]+1
        matchdata = json.loads(datastr[si:ei])
        return matchdata
    
    
    sind = [i for i,s in enumerate(sqlline) if s=='('];
    eind = np.array([i for i,s in enumerate(sqlline) if s==')'])+1;
    matches = []
    for i,ss in enumerate(sind):
        ee = eind[i]
        datastr = sqlline[ss:ee]
        matches.append(into_json(datastr))
    return matches

def load_sql_insert_file_data(sqlfile_str):
    """
    Parse the sql insert script.
    Load each line and parse the data into a list of dictionaries/list in json format.
    Each object in the list contains the information of one match.
    Input:
        sqlfile_str: The filename of the sql insert script
    Output:
        data: a list with each match as an object.
    """
    with open(sqlfile_str,'r') as f:
        data = f.readlines()
    linewithdata = [len(i)>1000 for i in data]
    data = np.array(data)[linewithdata]
    matches=[]
    for sqlline in data:
        matches = matches + parse_sql_line(sqlline)
    return matches
