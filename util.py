import numpy as np
import copy

def hand_into_vector(handlist):
    """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
    dict['Shape']: 
        Spades = 0,
        Hearts = 1,
        Diamond = 2,
        Clubs = 3,
        MahJong = 4,
        Dogs = 5,
        Dragon = 6,
        Phoinex = 7
    dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
    """
    handvec = np.zeros(56)
    for card in handlist:
        # If card is a regular card
        if card['Shape'] in range(0,4):
            ind = card['Shape']*13+card['Value']-2
        else:
            ind = 52 + card['Shape'] - 4
        handvec[ind] = 1
    return handvec

def vector_into_hand(handvec):
    """The reverse function from handvec into handlist. """
    handlist = []
    for ind, b in enumerate(handvec):
        card={}
        if b:
            if ind < 52:
                card['Shape'] = ind//13
                card['Value'] = ind%13 +2
            elif ind==52:
                card['Shape'] = 4
                card['Value'] = 1
            else:
                card['Shape'] = ind - 52 +4
                card['Value'] = 0
            handlist.append(card)
    return handlist


class grandData():
    """store the data with grand tichus in a more convenient format."""
    def __init__(self):
        self.cards8=[]
        self.cards6=[]
        self.made=[]
        self.playerlevel=[]

        
    def loaddata(self, matches,levelcutoff = 0):
        """Recurse through all the game/player combos. If grandtichu is called, load the data """
        teams = ['FirstTeam','SecondTeam']
        for match in matches:
            for r in match['Rounds']:
                for team in teams:
                    for p in r[team]['Players']:
                        if p['CalledGrand'] & (p['FinishOrder']!=0) & (p['Level'] >= levelcutoff):
                            self.cards8.append(self.hand2vec(p['Cards8']))
                            self.cards6.append(self.hand2vec(p['Cards6']))
                            self.playerlevel.append(p['Level'])
                            if p['FinishOrder'] == 1:
                                self.made.append(1)
                            else:
                                self.made.append(0)
        self.cards8 = np.array(self.cards8)
        self.cards6 = np.array(self.cards6)
        self.made = np.array(self.made)
        self.playerlevel = np.array(self.playerlevel)
        self.find_feature_vector()
        return

    def find_feature_vector(self):
        """
        From self.allcards, compute the feature vectors and add to self.feature_vector and self.feature_vector_extend
        """
        # One example to set the dimension
        hd = handDataGrand(self.cards8[0,:])
        fv = np.zeros([len(self.made),len(hd.feature_vector)])
        fv_extend = np.zeros([len(self.made),len(hd.feature_vector_extend)])
        compress_hand = np.zeros([len(self.made),len(hd.compress_hand)])
        ultimate = np.zeros([len(self.made),len(hd.ultimate)])
        # Loop over games 
        for i, handvec in enumerate(self.cards8):
            hd = handDataGrand(handvec)
            fv[i] = hd.feature_vector
            fv_extend[i] = hd.feature_vector_extend
            compress_hand[i] = hd.compress_hand
            ultimate[i] = hd.ultimate
        self.feature_vector = fv
        self.feature_vector_extend = fv_extend
        self.compress_hand = compress_hand
        self.feature_explanation = hd.feature_explanation
        self.ultimate = ultimate
    

    def hand2vec(self, handlist):
        """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
        dict['Shape']: 
            Spades = 0,
            Hearts = 1,
            Diamond = 2,
            Clubs = 3,
            MahJong = 4,
            Dogs = 5,
            Dragon = 6,
            Phoinex = 7
        dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
        """
        handvec = np.zeros(56)
        for card in handlist:
            # If card is a regular card
            if card['Shape'] in range(0,4):
                ind = card['Shape']*13+card['Value']-2
            else:
                ind = 52 + card['Shape'] - 4
            handvec[ind] = 1
        return handvec
        
    def vec2hand(self, handvec):
        """The reverse function from handvec into handlist. """
        handlist = []
        for ind, b in enumerate(handvec):
            card={}
            if b:
                if ind < 52:
                    card['Shape'] = ind//13
                    card['Value'] = ind%13 +2
                elif ind==52:
                    card['Shape'] = 4
                    card['Value'] = 1
                else:
                    card['Shape'] = ind - 52 +4
                    card['Value'] = 0
                handlist.append(card)
        return handlist

class tichuData():
    """store the data with grand tichus in a more convenient format."""
    def __init__(self):
        self.allcards=[]
        self.matecard=[]
        self.made=[]

    def loaddata(self, matches):
        """Recurse through all the game/player combos. If grandtichu is called, load the data """
        teams = ['FirstTeam','SecondTeam']
        for im, match in enumerate(matches):
            for ir,r in enumerate(match['Rounds']):
                for team in teams:
                    for ip,p in enumerate(r[team]['Players']):
                        if p['CalledTichu'] & (p['FinishOrder']!=0):
                            giveout = [p['RightCardGiven'], p['MateCardGiven'], p['LeftCardGiven']]
                            pos = p['Position']
                            receive = []
                            otherps = r[self.otherteam(team)]['Players']
                            for op in otherps:
                                if (op['Position']-1)%4 == p['Position']:
                                    if op['LeftCardGiven'] == None:
                                        print 'match {:d}, round {:d}'.format(im, ir)
                                        raise ValueError('why is it None?')
                                    # this is the player on the right
                                    receive.append(op['LeftCardGiven'])
                                else:
                                    if op['LeftCardGiven'] == None:
                                        print 'match {:d}, round {:d}'.format(im, ir)
                                        raise ValueError('why is it None?')
                                    # this is the player on the left
                                    receive.append(op['RightCardGiven'])
                            # this is the teammate
                            matecard = r[team]['Players'][1-ip]['MateCardGiven']
                            receive.append(matecard)
                            self.matecard.append(matecard)
                            # combine all the cards for the final hand
                            initvec = self.hand2vec(p['Cards8']+p['Cards6'])
                            givevec = self.hand2vec(giveout)
                            recvec = self.hand2vec(receive)
                            finalvec = initvec - givevec + recvec
                            self.allcards.append(finalvec)
                            # Record whether tichu's are made
                            if p['FinishOrder'] == 1:
                                self.made.append(1)
                            else:
                                self.made.append(0)
        self.allcards = np.array(self.allcards)
        self.made = np.array(self.made)
        self.find_feature_vector()
        return

    def find_feature_vector(self):
        """
        From self.allcards, compute the feature vectors and add to self.feature_vector and self.feature_vector_extend
        """
        # One example to set the dimension
        hd = handData(self.allcards[0,:])
        fv = np.zeros([len(self.made),len(hd.feature_vector)])
        fv_extend = np.zeros([len(self.made),len(hd.feature_vector_extend)])
        compress_hand = np.zeros([len(self.made),len(hd.compress_hand)])
        self.fv_plus_compress_hand = np.zeros([len(self.made),len(hd.fv_plus_compress_hand)])
        self.ultimate = np.zeros([len(self.made),len(hd.ultimate)])
        # Loop over games 
        for i, handvec in enumerate(self.allcards):
            hd = handData(handvec)
            fv[i] = hd.feature_vector
            fv_extend[i] = hd.feature_vector_extend
            compress_hand[i] = hd.compress_hand
            self.fv_plus_compress_hand[i] = hd.fv_plus_compress_hand
            self.ultimate[i] = hd.ultimate
        self.feature_vector = fv
        self.feature_vector_extend = fv_extend
        self.compress_hand = compress_hand
        self.feature_explanation = hd.feature_explanation

    
    def otherteam(self, team):
        if team=='FirstTeam':
            return 'SecondTeam'
        elif team=='SecondTeam':
            return 'FirstTeam'
        else:
            raise ValueError('unknown team name')
    def opponent_tichu_condition(self, rounddata, team, ip):
        """
        See if the team satisfies a completing hand condition as the opponent
        """
        ourteam = rounddata[team]['Players']
        oppteam =  rounddata[self.otherteam(team)]['Players']
        if ourteam[0]['CalledTichu'] + ourteam[1]['CalledTichu'] == 0:  # this team does not call tichu
            # opponent calls tichu
            if oppteam[0]['CalledTichu'] + oppteam[1]['CalledTichu'] > 0: 
                # teammate does not go out first
                if ourteam[1-ip]['FinishOrder'] != 1:
                    return True
        return False
    
    def hand2vec(self, handlist):
        """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
        dict['Shape']: 
            Spades = 0,
            Hearts = 1,
            Diamond = 2,
            Clubs = 3,
            MahJong = 4,
            Dogs = 5,
            Dragon = 6,
            Phoinex = 7
        dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
        """
        handvec = np.zeros(56)
        for card in handlist:
            # If card is a regular card
            if card['Shape'] in range(0,4):
                ind = card['Shape']*13+card['Value']-2
            else:
                ind = 52 + card['Shape'] - 4
            handvec[ind] = 1
        return handvec
        
    def vec2hand(self, handvec):
        """The reverse function from handvec into handlist. """
        handlist = []
        for ind, b in enumerate(handvec):
            card={}
            if b:
                if ind < 52:
                    card['Shape'] = ind//13
                    card['Value'] = ind%13 +2
                elif ind==52:
                    card['Shape'] = 4
                    card['Value'] = 1
                else:
                    card['Shape'] = ind - 52 +4
                    card['Value'] = 0
                handlist.append(card)
        return handlist

class tichuDataExtend():
    """store the data with tichus in a more convenient format.
    To get more data, I add in the hands of players when their opponents call tichu
    Case: Opponent calls tichu, and teammate did not go out first.

    """
    def __init__(self):
        self.allcards=[]
        self.matecard=[]
        self.made=[]
        
    def loaddata(self, matches):
        """Recurse through all the game/player combos. If grandtichu is called, load the data """
        teams = ['FirstTeam','SecondTeam']
        for im, match in enumerate(matches):
            for ir,r in enumerate(match['Rounds']):
                for team in teams:
                    for ip,p in enumerate(r[team]['Players']):
                        # consider the case where opponent called tichu and they did not call tichu
                        oppcond = self.opponent_tichu_condition(r, team, ip)
                        if (p['CalledTichu'] | oppcond) & (p['FinishOrder']!=0):
                            giveout = [p['RightCardGiven'], p['MateCardGiven'], p['LeftCardGiven']]
                            pos = p['Position']
                            receive = []
                            otherps = r[self.otherteam(team)]['Players']
                            for op in otherps:
                                if (op['Position']-1)%4 == p['Position']:
                                    if op['LeftCardGiven'] == None:
                                        print 'match {:d}, round {:d}'.format(im, ir)
                                        raise ValueError('why is it None?')
                                    # this is the player on the right
                                    receive.append(op['LeftCardGiven'])
                                else:
                                    if op['LeftCardGiven'] == None:
                                        print 'match {:d}, round {:d}'.format(im, ir)
                                        raise ValueError('why is it None?')
                                    # this is the player on the left
                                    receive.append(op['RightCardGiven'])
                            # this is the teammate
                            matecard = r[team]['Players'][1-ip]['MateCardGiven']
                            receive.append(matecard)
                            self.matecard.append(matecard)
                            # combine all the cards for the final hand
                            initvec = self.hand2vec(p['Cards8']+p['Cards6'])
                            givevec = self.hand2vec(giveout)
                            recvec = self.hand2vec(receive)
                            finalvec = initvec - givevec + recvec
                            self.allcards.append(finalvec)
                            # Record whether tichu's are made
                            if p['FinishOrder'] == 1:
                                self.made.append(1)
                            else:
                                self.made.append(0)
        self.allcards = np.array(self.allcards)
        self.made = np.array(self.made)
        self.find_feature_vector()
        return

    def find_feature_vector(self):
        """
        From self.allcards, compute the feature vectors and add to self.feature_vector and self.feature_vector_extend
        """
        # One example to set the dimension
        hd = handData(self.allcards[0,:])
        fv = np.zeros([len(self.made),len(hd.feature_vector)])
        fv_extend = np.zeros([len(self.made),len(hd.feature_vector_extend)])
        compress_hand = np.zeros([len(self.made),len(hd.compress_hand)])
        self.fv_plus_compress_hand = np.zeros([len(self.made),len(hd.fv_plus_compress_hand)])
        self.ultimate = np.zeros([len(self.made),len(hd.ultimate)])
        # Loop over games 
        for i, handvec in enumerate(self.allcards):
            hd = handData(handvec)
            fv[i] = hd.feature_vector
            fv_extend[i] = hd.feature_vector_extend
            compress_hand[i] = hd.compress_hand
            self.fv_plus_compress_hand[i] = hd.fv_plus_compress_hand
            self.ultimate[i] = hd.ultimate
        self.feature_vector = fv
        self.feature_vector_extend = fv_extend
        self.compress_hand = compress_hand
        self.feature_explanation = hd.feature_explanation

    
    def otherteam(self, team):
        if team=='FirstTeam':
            return 'SecondTeam'
        elif team=='SecondTeam':
            return 'FirstTeam'
        else:
            raise ValueError('unknown team name')
    def opponent_tichu_condition(self, rounddata, team, ip):
        """
        See if the team satisfies a completing hand condition as the opponent
        """
        ourteam = rounddata[team]['Players']
        oppteam =  rounddata[self.otherteam(team)]['Players']
        if ourteam[0]['CalledTichu'] + ourteam[1]['CalledTichu'] == 0:  # this team does not call tichu
            # opponent calls tichu
            if oppteam[0]['CalledTichu'] + oppteam[1]['CalledTichu'] > 0: 
                # teammate does not go out first
                if ourteam[1-ip]['FinishOrder'] != 1:
                    return True
        return False
    
    def hand2vec(self, handlist):
        """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
        dict['Shape']: 
            Spades = 0,
            Hearts = 1,
            Diamond = 2,
            Clubs = 3,
            MahJong = 4,
            Dogs = 5,
            Dragon = 6,
            Phoinex = 7
        dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
        """
        handvec = np.zeros(56)
        for card in handlist:
            # If card is a regular card
            if card['Shape'] in range(0,4):
                ind = card['Shape']*13+card['Value']-2
            else:
                ind = 52 + card['Shape'] - 4
            handvec[ind] = 1
        return handvec
        
    def vec2hand(self, handvec):
        """The reverse function from handvec into handlist. """
        handlist = []
        for ind, b in enumerate(handvec):
            card={}
            if b:
                if ind < 52:
                    card['Shape'] = ind//13
                    card['Value'] = ind%13 +2
                elif ind==52:
                    card['Shape'] = 4
                    card['Value'] = 1
                else:
                    card['Shape'] = ind - 52 +4
                    card['Value'] = 0
                handlist.append(card)
        return handlist


class handData():
    """
    Playable cards pattern analysis. Analyze the 14 card hand for:
        Bombs, threes, pairs, straights, tractors, singleton small card.
    Use these to make a feature vector.

    """
    def __init__(self, handvec):
        self.handvec = handvec
        self.hand = self.vec2hand(handvec)
        self.vlist = self.compute_numbers(self.hand)
        self.find_straights()
        self.find_three_of_a_kind()
        self.find_pairs()
        self.find_bombs()
        self.find_single_small_card()
        self.count_big_cards()
        self.find_tractors()
        self.create_feature_vector()

    def compute_numbers(self, hand):
        """
        given hand (list of dictionary), compute the number of cards in each value
        """
        valuelist = {}
        for v in range(1,15):
            valuelist[v] = 0
        for c in hand:
            if c['Value'] > 0:
                valuelist[c['Value']] += 1
        return valuelist
        
    def find_straights(self):
        straightlength = {}
        for sv in range(1,11): #largest straight 10,J,Q,K,A
            straightlength[sv] = 0
            for v in range(sv,15):
                if self.vlist[v] > 0:
                    straightlength[sv] += 1
                else:
                    break
        self.longest_straight_length = max(straightlength.values())
        # find the longest straight length
        if self.longest_straight_length < 5:
            self.longest_straight_length = 0
        allstraights = [k for k in straightlength.keys() if straightlength[k]>=5]
        # if no straights, set default largest_straight to -1
        self.largest_straight = -1
        if len(allstraights) > 0:
            self.largest_straight = max(allstraights)
        self.all_straight_length = straightlength
        
    def find_three_of_a_kind(self):
        num = 0
        # what should I set the value to if the hand does not have three of a kind?
        largest = -1
        threes = []
        for v in self.vlist.keys():
            if self.vlist[v] == 3:
                num += 1
                threes.append(v)
                if v > largest:
                    largest = v
        self.num_of_threes = num
        self.largest_three = largest
        self.threes = np.array(threes)
        
    def find_pairs(self):
        num = 0
        # what should I set the value to if the hand does not have three of a kind?
        largest = -1
        pairs = []
        for v in self.vlist.keys():
            if self.vlist[v] == 2:
                num += 1
                pairs.append(v)
                if v > largest:
                    largest = v
        self.num_of_pairs = num
        self.largest_pair = largest
        self.pairs = np.array(pairs)
        
    def find_bombs(self):
        num = 0
        fours = []
        largest = 0
        cards_in_bomb = []
        # find 4 of a kind
        for v in self.vlist.keys():
            if self.vlist[v] == 4:
                num += 1
                fours.append(v)
                if v > largest:
                    largest = v
        # find straight flush
        self.sf_suit = -1
        self.sf_range = []
        # organize into suits
        regular_card_array = np.zeros([4,15]);
        for c in self.hand:
            if c['Shape'] in range(4):
                regular_card_array[c['Shape'], c['Value']] = 1
        # detect flush
        for i,suit in enumerate(regular_card_array):
            # pad zeros in front and back
            padsuit = np.concatenate(([0], suit, [0]))
            absdiff = np.abs(np.diff(padsuit))
            ranges = np.where(absdiff == 1)[0].reshape(-1,2)
            isstraightflush = np.array([(r[1]-r[0] >= 5) for r in  ranges])
            for j,b in enumerate(isstraightflush):
                if b:
                    num += 1
                    # I'll just record the last one detected lol
                    self.sf_suit = i
                    self.sf_range = ranges[j]
                    # count -1 if the cards are the same as the 4 of kind bomb
                    for f in fours:
                        if f < ranges[j][1] & f > ranges[j][0]:
                            num -= 1
            
        self.num_of_bombs = num
        self.fours = np.array(fours)
    
    def find_single_small_card(self):
        # define small cards as cards smaller and equal to 10 
        # that are not in pairs, straights, threes, and bombs.
        
        ### Method1: remove pairs, threes, fours and then straights
        leftover = []
        for c in self.hand:
            if c['Shape']<=4 :
                if self.vlist[c['Value']] == 1:
                    leftover.append(c)
        # remove straights if exist in leftover
        vlist = self.compute_numbers(leftover)
        done = 0
        while not done:
            vlist, done = self.remove_straight(vlist)
        for k in vlist.keys():
            # take value <= J as small cards 
            if k > 11:
                vlist[k] = 0
        self.leftover1 = vlist
        self.small_card1 = sum(vlist.values())

        ### Method2: remove straights, then pairs, threes, fours
        vlist = self.compute_numbers(self.hand)
        # remove straights 
        done = 0
        while not done:
            vlist, done = self.remove_straight(vlist)
        for k in vlist.keys():
            # take value <= J as small cards 
            if (k > 11) | (vlist[k] > 1):
                vlist[k] = 0
        self.leftover2 = vlist
        self.small_card2 = sum(vlist.values())
        self.num_small_card = min(self.small_card1, self.small_card2)

    def remove_straight(self, vlist):
        """
        detect cards in straights given a handlist
        For simplicity, remove the longest straight and then recurse
        """
        straightlength = np.zeros(11);
        for sv in range(1,11): #largest straight 10,J,Q,K,A
            for v in range(sv,15):
                if vlist[v] > 0:
                    straightlength[sv] += 1
                else:
                    break
        longest = int(max(straightlength))
        if longest >= 5:
            longidx = np.argmax(straightlength)
            for ii in range(longidx, longidx+longest):
                vlist[ii] -= 1
            done = 0
        else:
            done = 1
        return vlist, done
    
    def count_big_cards(self):
        self.ace = self.vlist[14]
        self.phoenix = 0
        self.dragon = 0
        self.dog = 0
        if {'Shape': 7, 'Value': 0} in self.hand:
            self.phoenix += 1
        if {'Shape': 6, 'Value': 0} in self.hand:
            self.dragon += 1
        if {'Shape': 5, 'Value': 0} in self.hand:
            self.dog += 1
        
    def find_tractors(self):
        """
        call this after self.find_pairs() is done
        """
        self.num_of_tractors = 0
        for p in self.pairs:
            if p+1 in self.pairs:
                self.num_of_tractors += 1
                
    def hand2vec(self, handlist):
        """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
        dict['Shape']: 
            Spades = 0,
            Hearts = 1,
            Diamond = 2,
            Clubs = 3,
            MahJong = 4,
            Dogs = 5,
            Dragon = 6,
            Phoinex = 7
        dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
        """
        handvec = np.zeros(56)
        for card in handlist:
            # If card is a regular card
            if card['Shape'] in range(0,4):
                ind = card['Shape']*13+card['Value']-2
            else:
                ind = 52 + card['Shape'] - 4
            handvec[ind] = 1
        return handvec
        
    def vec2hand(self, handvec):
        """The reverse function from handvec into handlist. """
        handlist = []
        for ind, b in enumerate(handvec):
            card={}
            if b:
                if ind < 52:
                    card['Shape'] = ind//13
                    card['Value'] = ind%13 +2
                elif ind==52:
                    card['Shape'] = 4
                    card['Value'] = 1
                else:
                    card['Shape'] = ind - 52 +4
                    card['Value'] = 0
                handlist.append(card)
        return handlist
    
    def create_feature_vector(self):
        """
        Combine all the patterns that I identified into a feature vector.
            self.find_straights()
            self.find_three_of_a_kind()
            self.find_pairs()
            self.find_bombs()
            self.find_single_small_card()
            self.count_big_cards()
            self.find_tractors()
        feature_vector = [ace, dog, dragon, phoenix, num_of_bombs, num_of_threes, largest_threes, num_of_pairs, largest_pair,
                         longest_straight_length, largest_straight, num_of_tractors,  num_small_cards]
        feature_vector_extend = [feature_vector, 54 cards]
        compressed = [counts of 1 to 14, dog, dragon, phoenix, num_of_straightflush]
        """
        self.feature_vector = np.array([self.ace, self.dog, self.dragon, self.phoenix, self.num_of_bombs, \
                               self.num_of_threes, self.largest_three, \
                               self.num_of_pairs, self.largest_pair, \
                               self.longest_straight_length, self.largest_straight, \
                               self.num_of_tractors,  self.num_small_card ])
        
        self.feature_vector_extend = np.concatenate((self.feature_vector, self.handvec))
        
        self.feature_explanation = ['self.ace', 'self.dog', 'self.dragon', 'self.phoenix', 'self.num_of_bombs', \
                          'self.num_of_threes', 'self.largest_three', \
                          'self.num_of_pairs', 'self.largest_pair', \
                          'self.longest_straight_length', 'self.largest_straight', \
                          'self.num_of_tractors',  'self.num_small_card']
        ##### create a compressed vector
        compress_hand = np.zeros(18)
        for k in self.vlist.keys():
            compress_hand[k-1] = self.vlist[k]
        # dog
        if {'Shape':5, 'Value':0} in self.hand:
            compress_hand[14] = 1
        # dragon
        if {'Shape':6, 'Value':0} in self.hand:
            compress_hand[15] = 1
        # phoenix
        if {'Shape':7, 'Value':0} in self.hand:
            compress_hand[16] = 1
        # straight flush
        if self.sf_suit != -1:
            compress_hand[17] = 1
        self.compress_hand = compress_hand
        
        #### create a feature+compress hand vector
        self.fv_plus_compress_hand = np.concatenate((self.feature_vector, self.compress_hand))
        
        ### compute the one number statistics
        self.ultimate = self.reduce_tichu()

    def reduce_tichu(self):
        selectfeature = np.array([0,1,2,3,4,9,12])
        temp = copy.deepcopy(self.feature_vector)
        temp[9] = temp[9] > 0
        w = np.array([2,-2,6,6,5,1,-1])
        return np.array([1,sum(temp[selectfeature]*w)])




class handDataGrand():
    def __init__(self, handvec):
        self.handvec = handvec
        self.hand = self.vec2hand(handvec)
        self.vlist = self.compute_numbers(self.hand)
        self.find_straights()
        self.find_three_of_a_kind()
        self.find_pairs()
        self.find_bombs()
        self.find_single_small_card()
        self.count_big_special_cards()
        self.find_tractors()
        self.create_feature_vector()

    def compute_numbers(self, hand):
        """
        given hand (list of dictionary), compute the number of cards in each value
        """
        valuelist = {}
        for v in range(1,15):
            valuelist[v] = 0
        for c in hand:
            if c['Value'] > 0:
                valuelist[c['Value']] += 1
        return valuelist

    def find_straights(self):
        straightlength = {}
        for sv in range(1,11): #largest straight 10,J,Q,K,A
            straightlength[sv] = 0
            for v in range(sv,15):
                if self.vlist[v] > 0:
                    straightlength[sv] += 1
                else:
                    break
        self.longest_consecutive = max(straightlength.values())
        self.straight = 0
        if self.longest_consecutive >= 5:
            self.straight = 1

    def find_three_of_a_kind(self):
        num = 0
        # what should I set the value to if the hand does not have three of a kind?
        largest = -1
        threes = []
        for v in self.vlist.keys():
            if self.vlist[v] == 3:
                num += 1
                threes.append(v)
                if v > largest:
                    largest = v
        self.num_of_threes = num
        self.largest_three = largest
        self.threes = np.array(threes)

    def find_pairs(self):
        num = 0
        # what should I set the value to if the hand does not have three of a kind?
        largest = -1
        pairs = []
        for v in self.vlist.keys():
            if self.vlist[v] == 2:
                num += 1
                pairs.append(v)
                if v > largest:
                    largest = v
        self.num_of_pairs = num
        self.largest_pair = largest
        self.pairs = np.array(pairs)

    def find_bombs(self):
        num = 0
        fours = []
        largest = 0
        cards_in_bomb = []
        # find 4 of a kind
        for v in self.vlist.keys():
            if self.vlist[v] == 4:
                num += 1
                fours.append(v)
                if v > largest:
                    largest = v
        # find straight flush
        self.sf_suit = -1
        self.sf_range = []
        # organize in to suits
        regular_card_array = np.zeros([4,15]);
        for c in self.hand:
            if c['Shape'] in range(4):
                regular_card_array[c['Shape'], c['Value']] = 1
        # detect flush
        for i,suit in enumerate(regular_card_array):
            # pad zeros in front and back
            padsuit = np.concatenate(([0], suit, [0]))
            absdiff = np.abs(np.diff(padsuit))
            ranges = np.where(absdiff == 1)[0].reshape(-1,2)
            isstraightflush = np.array([(r[1]-r[0] >= 5) for r in  ranges])
            for j,b in enumerate(isstraightflush):
                if b:
                    num += 1
                    # I'll just record the last one detected lol
                    self.sf_suit = i
                    self.sf_range = ranges[j]
                    # count -1 if the cards are the same as the 4 of kind bomb
                    for f in fours:
                        if f < ranges[j][1] & f > ranges[j][0]:
                            num -= 1
        self.num_of_bombs = num
        self.fours = np.array(fours)

    def find_single_small_card(self):
        # define small cards as cards smaller and equal to 10
        # that are not in pairs, straights, threes, and bombs.
        ### Method1: remove pairs, threes, fours and then straights
        leftover = []
        for c in self.hand:
            if c['Shape']<=4 :
                if self.vlist[c['Value']] == 1:
                    leftover.append(c)
        # remove straights if exist in leftover
        vlist = self.compute_numbers(leftover)
        done = 0
        while not done:
            vlist, done = self.remove_straight(vlist)
        for k in vlist.keys():
            if k > 11 or k ==1:
                vlist[k] = 0
        self.leftover1 = vlist
        self.small_card1 = sum(vlist.values())

        ### Method2: remove straights, then pairs, threes, fours
        vlist = self.compute_numbers(self.hand)
        # remove straights
        done = 0
        while not done:
            vlist, done = self.remove_straight(vlist)
        for k in vlist.keys():
            if (k > 11) | (vlist[k] > 1) | (k==1):
                vlist[k] = 0
        self.leftover2 = vlist
        self.small_card2 = sum(vlist.values())
        self.num_small_card = min(self.small_card1, self.small_card2)


    def remove_straight(self, vlist):
        """
        detect cards in straights given a handlist
        For simplicity, remove the longest straight and then recurse
        """
        straightlength = np.zeros(11);
        for sv in range(1,11): #largest straight 10,J,Q,K,A
            for v in range(sv,15):
                if vlist[v] > 0:
                    straightlength[sv] += 1
                else:
                    break
        longest = int(max(straightlength))
        if longest >= 5:
            longidx = np.argmax(straightlength)
            for ii in range(longidx, longidx+longest):
                vlist[ii] -= 1
            done = 0
        else:
            done = 1
        return vlist, done

    def count_big_special_cards(self):
        """
        Count numbers of ace,
        whether hand contains dragon, phoenix
        """
        self.ace = self.vlist[14]
        self.phoenix = 0
        self.dragon = 0
        self.dog = 0
        self.mahjong = 0
        if {'Shape': 7, 'Value': 0} in self.hand:
            self.phoenix = 1
        if {'Shape': 6, 'Value': 0} in self.hand:
            self.dragon = 1
        if {'Shape': 4, 'Value': 1} in self.hand:
            self.mahjong = 1
        if {'Shape': 5, 'Value': 0} in self.hand:
            self.dog = 1
    
    def find_tractors(self):
        """
        call this after self.find_pairs() is done
        """
        self.num_of_tractors = 0
        for p in self.pairs:
            if p+1 in self.pairs:
                self.num_of_tractors += 1

    def hand2vec(self, handlist):
        """handlist is a list of dictionary, each dict has dict['Shape'] and dict['Value']
        dict['Shape']:
            Spades = 0,
            Hearts = 1,
            Diamond = 2,
            Clubs = 3,
            MahJong = 4,
            Dogs = 5,
            Dragon = 6,
            Phoinex = 7
        dict['Value']: card number (2-14). 14 for Ace. 0 for Dogs, Dragon, Phoinex. MahJong has value = 1
        """
        handvec = np.zeros(56)
        for card in handlist:
            # If card is a regular card
            if card['Shape'] in range(0,4):
                ind = card['Shape']*13+card['Value']-2
            else:
                ind = 52 + card['Shape'] - 4
            handvec[ind] = 1
        return handvec

    def vec2hand(self, handvec):
        """The reverse function from handvec into handlist. """
        handlist = []
        for ind, b in enumerate(handvec):
            card={}
            if b:
                if ind < 52:
                    card['Shape'] = ind//13
                    card['Value'] = ind%13 +2
                elif ind==52:
                    card['Shape'] = 4
                    card['Value'] = 1
                else:
                    card['Shape'] = ind - 52 +4
                    card['Value'] = 0
                handlist.append(card)
        return handlist

    def create_feature_vector(self):
        """
        Combine all the patterns that I identified into a feature vector.
            self.find_straights()
            self.find_three_of_a_kind()
            self.find_pairs()
            self.find_bombs()
            self.find_single_small_card()
            self.count_big_cards()
            self.find_tractors()
        feature_vector = [num_big_cards, num_of_bombs, num_of_threes, largest_threes, num_of_pairs, largest_pair,
                         longest_straight_length, largest_straight, num_of_tractors,  num_small_cards]
        """
        self.feature_explanation = ['self.ace', 'self.phoenix', 'self.dragon', 'self.dog', 'self.mahjong',\
                                    'self.num_of_bombs', \
                                    'self.num_of_threes', 'self.largest_three', \
                                    'self.num_of_pairs', 'self.largest_pair', \
                                    'self.longest_consecutive', 'self.straight', \
                                    'self.num_of_tractors',  'self.num_small_card']
        self.feature_vector = np.array([self.ace, self.phoenix, self.dragon, self.dog, self.mahjong,\
                               self.num_of_bombs, \
                               self.num_of_threes, self.largest_three, \
                               self.num_of_pairs, self.largest_pair, \
                               self.longest_consecutive, self.straight, \
                               self.num_of_tractors,  self.num_small_card ])

        # create a compressed vector
        compress_hand = np.zeros(18)
        for k in self.vlist.keys():
            compress_hand[k-1] = self.vlist[k]
        # dog
        if {'Shape':5, 'Value':0} in self.hand:
            compress_hand[14] = 1
        # dragon
        if {'Shape':6, 'Value':0} in self.hand:
            compress_hand[15] = 1
        # phoenix
        if {'Shape':7, 'Value':0} in self.hand:
            compress_hand[16] = 1
        # straight flush
        if self.sf_suit != -1:
            compress_hand[17] = 1
        self.compress_hand = compress_hand

        # add card counts into feature vector
        self.feature_vector_extend = np.concatenate((self.feature_vector, self.compress_hand[:14]))

        # compute the one number statistics
        self.ultimate = self.reduce_grand_tichu()

    def reduce_grand_tichu(self):
        selectfeature = np.array([0,1,2,5])
        w = np.array([1,3,3,3])
        return np.array([1,sum(self.feature_vector[selectfeature]*w)])
        
        
