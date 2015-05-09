"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Mon Oct 13 18:50:34 2014

@author: Wasit
"""
#scmaster.py
import os
import numpy as np
from IPython import parallel
class master:
    def __init__(self,dsetname='dataset_pickle'):
        n_proposal=10000
        self.dsetname=dsetname
        print("master>>init() dsetname: {}".format(dsetname))
#creat dview
        print("master>> create dview")
        # init cluster client
        self.clients = parallel.Client(packer='pickle')
        self.clients.block = True
        #0 use master as engine
        #1 donot use master as engine
        self.dview = self.clients.direct_view(self.clients.ids[2:])
        self.dview.block = True
#engine init
        self.dview.execute('import os; os.chdir("%s")'%os.getcwd())
        #self.eng=engine()
        print("master>> init engine")
        self.dview.execute('from %s import dataset'%dsetname)
        for i,dv in enumerate(self.clients):
            dv.execute('dset=dataset(%d,%d)'\
            %(i,n_proposal//len(self.clients.ids)))
        self.dview.execute('from scengine import engine')
        self.dview.execute('eng=engine(dset)')
        self.engines_path=self.dview.gather('dset.path')
        print "debug:master:__init__: %s"%self.engines_path

#dont need to gather
#        print("master>> gather engines")        
#        self.engs=self.dview.gather('eng')
#        print("master>>engs:\n{}".format(self.engs))
#init local variables
        print("master>> init local variables")         
        self.minbagsize=2
        self.maxdepth=20
        #self.maxdepth=10        
        self.queue=None
        self.root=None
        self.node=None
    def __str__(self):
        paths=""
        for p in self.engines_path:
            paths=paths+"\n\t\t"+p
        return "\tminbagsize:%d maxdepth:%d engines_path:%s"%(self.minbagsize,self.maxdepth,paths)
    def __del__(self):
        pass
    def reset(self):
#engine reset        
        #H,Q = self.eng.reset()
        print("master>>reset()")
        self.dview.execute('H,Q = eng.reset()')
        Hs=np.array(self.dview['H'])
        Qs=np.array(self.dview['Q'])
        H=np.sum(Hs*Qs)/np.sum(Qs)
        Q=np.sum(Qs)        
        
        del self.root
        del self.queue
        del self.node
        self.root=mnode(self.maxdepth,H,Q)
        self.queue=[self.root]
        self.node=[]
        
        print("master>>reset() H: {:.4f}".format(H))
        print("master>>reset() Q: {:05d}".format(Q))
        
    def pop(self):
#engine pop
        #self.eng.pop()
        self.dview.execute('eng.pop()')
        self.node=self.queue.pop()
    def train(self,strtime='_time'):
        if not os.path.exists(self.dsetname):
            os.makedirs(self.dsetname)
        f1=open(self.dsetname+'/'+strtime+'.out', 'a')
        f1.write('>'*7+" {}".format(strtime)+'<'*7+'\n')
        while 0<len(self.queue):
            self.pop()
            self.search()
            
            f1.write("{}\n".format(self.node))
            
            #print self.node
        f1.write('-'*21+'\n')
        f1.close()
        self.clients.purge_results('all')        
        self.clients.purge_everything()
            
    def search(self):        
        #check depth
        if self.node.depth<=1:
            self.terminate('D')                
            return
#engine collect_thetas_taus
        print("master:search() collect_thetas_taus")
        #thetas,taus=self.eng.getParam()
        self.dview.execute('thetas,taus=eng.getParam()')
        #thetas=self.dview['thetas']
        #taus=self.dview['taus']

        
        #mearge_thetas_taus
        #all_thetas=np.concatenate(thetas,axis=1)
        #all_taus=np.concatenate(taus,axis=1)
        all_thetas=self.dview.gather('thetas')
        all_taus=self.dview.gather('taus')
#engine compute ensemble entropy
        print("master:search() compute entropy")
        #QH,Q=self.eng.getQH(all_thetas,all_taus)
        self.dview['all_thetas']=all_thetas
        self.dview['all_taus']=all_taus
        self.dview.execute('QH,Q=eng.getQH(all_thetas,all_taus)')
        QH=np.sum(np.array(self.dview['QH']),axis=0)
        Q=np.array(self.dview['Q'])#+np.finfo(np.float32).tiny
        
        #check bag size Q
        print("master:search() check bag size")
        if np.max(Q)<self.minbagsize:
            self.terminate('Q')                
            return
        #compute the entropy gain
        print("master:search() compute entropy gain")
        gain=self.node.H-QH/(np.sum(Q))
        #find best gain index
        bgi = np.argmax(gain)
        #check gain
        if gain[bgi]<np.finfo(np.float32).tiny:
            self.terminate('G')                
            return
#engine split
        print("master:search() engine split")
        best_theta=all_thetas[bgi,:]
        best_tau=all_taus[bgi]
        self.dview['best_theta']=best_theta
        self.dview['best_tau']=best_tau
        #print("best theta: {},tau: {}".format(best_theta,best_tau))
        print("master:search() eng.split()")
        self.dview.execute('HL,QL,HR,QR = eng.split(best_theta,best_tau)')
        HLs=np.array(self.dview['HL'])
        QLs=np.array(self.dview['QL'])
        HRs=np.array(self.dview['HR'])
        QRs=np.array(self.dview['QR'])
        
        HL=np.sum(HLs*QLs)/np.sum(QLs)
        QL=np.sum(QLs)
        
        HR=np.sum(HRs*QRs)/np.sum(QRs)
        QR=np.sum(QRs)
        
        print("master:search() append_nodes")
        self.node.L=mnode(self.node.depth-1,HL,QL,'L',self.node)
        self.node.R=mnode(self.node.depth-1,HR,QR,'R',self.node)
        self.queue.append(self.node.L)        
        self.queue.append(self.node.R)
        
        self.node.theta=all_thetas[bgi,:]
        self.node.tau=all_taus[bgi]
        self.node.char=self.node.char+'-'        
        
    def terminate(self, code):
        self.node.char=self.node.char+code
#engine getHist
        #P=self.eng.getHist()#+np.finfo(np.float32).tiny
        self.dview.execute('P=eng.getHist()')
        P=np.sum(np.array(self.dview['P']),axis=0)
        self.node.P=P/np.sum(P)

class mnode:
    def __init__(self,depth=0,H=0,Q=0,char='*',parent=None):
        self.theta=None  #vector array
        self.tau=None  #scalar
        self.H=H  #scalar
        self.P=None  #vector array
        self.parent=parent  #mnode
        self.depth=depth  #int
        self.L=None  #mnode
        self.R=None  #mnode
        self.char=char
        self.Q=Q
    def __del__(self):
        del self.theta
        del self.tau
        del self.H
        del self.P
        del self.parent
        del self.depth
        del self.L
        del self.R
        del self.char
    def table(self):
        text = self.__repr__()+'\n'
        if self.L is not None:
            text=text+self.L.table()
        if self.R is not None:
            text=text+self.R.table()
 
        return text
    
    def __repr__(self):
        #np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        if self.tau is None:
            ids = self.P.argsort()[::-1][:3]
            string='%s %02d H:%.3e,Q:%06d (cl,P):(%03d,%.2f) (%03d,%.2f) (%03d,%.2f)' % (
            self.char,self.depth,self.H,self.Q,ids[0],self.P[ids[0]],ids[1],self.P[ids[1]],ids[2],self.P[ids[2]])
        else:
            string='%s %02d H:%.3e,Q:%06d tau:%s' % (self.char,self.depth,self.H,self.Q,self.tau)
        return string

if __name__ == '__main__':
    m=master()
    m.reset()
    m.train()
    
    import pickle
    pickleFile = open('root.pic', 'wb')
    pickle.dump(m.root, pickleFile, pickle.HIGHEST_PROTOCOL)
    pickleFile.close()



