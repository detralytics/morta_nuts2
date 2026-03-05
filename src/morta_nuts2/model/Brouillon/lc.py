import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.special import gammaln
from scipy.linalg import lstsq


# =============================================================================
# 1. CONSTRUCTION construction du modèles Lee carter : inspiré de Donatien
# =============================================================================


def LC_fit(ax, bx,kappa,Extg,Dxtg,xv,tv,nb_iter):
    #gradient descent parameter
    eta = 1
    for it in range(nb_iter):
        for ct_opt in np.arange(0,3):
            ax = ax.reshape(-1,1) ; bx = bx.reshape(-1,1) 
            nb_regions = Extg.shape[2]
            axM     = np.repeat(ax,len(tv),axis=1)
            bxM     = np.repeat(bx,len(tv),axis=1)            
            kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
            logmuxt_baseline = axM+bxM*kappaM
            logmuxt_grp  = np.zeros((len(xv),len(tv),nb_regions))    
            #computation of log(mu(x,t,g))
            for ct in range(nb_regions):                
                logmuxt_grp[:,:,ct] = logmuxt_baseline.copy()
            #baseline for update
            dlnL_baseline  = (Dxtg - Extg*np.exp(logmuxt_grp))
            if (ct_opt==0):
                #--------------- ax --------------------
                ax_new    = np.zeros_like(ax)             
                dlnL_dpar = (np.sum(dlnL_baseline,axis=(1,2))/
                            np.sum(Extg*np.exp(logmuxt_grp),axis=(1,2)) )
                ax_new = ax + eta* dlnL_dpar.reshape(-1,1)                
                #update
                ax     = ax_new.copy()            
            if (ct_opt==1):
                #--------------- bx --------------------
                bx_new = np.zeros_like(bx)            
                kappaM= np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
                kappaM= np.expand_dims(kappaM,axis=2)
                kappaM= np.repeat(kappaM,nb_regions,axis=2)                
                dlnL_dpar = (np.sum(dlnL_baseline*kappaM,axis=(1,2))/(
                            np.sum(Extg*np.exp(logmuxt_grp)*kappaM**2,axis=(1,2))))
                bx_new = bx + eta*dlnL_dpar.reshape(-1,1)     
                #we normalize
                scal_bx   = np.sum(bx_new)
                bx_new    = bx_new /scal_bx
                kappa     = kappa*scal_bx
                bx        = bx_new.copy()                
            if (ct_opt==2):
                #---------------Kappa-----------------    
                # warning we use the old betax(x)
                kappa_new = np.zeros_like(kappa)
                bxM = np.repeat(bx,len(tv),axis=1)
                bxM = np.expand_dims(bxM,axis=2)
                bxM = np.repeat(bxM,nb_regions,axis=2)                
                dlnL_dpar = (np.sum(dlnL_baseline*bxM,axis=(0,2))/(
                      np.sum(Extg*np.exp(logmuxt_grp)*bxM**2,axis=(0,2))))
                kappa_new = kappa + eta*dlnL_dpar
                #we rescale
                kappa_avg = np.mean(kappa_new)
                kappa_new = (kappa_new - kappa_avg) #*np.sum(bx)
                ax        =  ax + kappa_avg*bx               
                #update
                kappa  = kappa_new.copy()         
    #end loop
    # we recompute log-mort. rates
    ax = ax.reshape(-1,1)  ;  bx = bx.reshape(-1,1)         
    axM     = np.repeat(ax,len(tv),axis=1)
    bxM     = np.repeat(bx,len(tv),axis=1)                
    kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
    logmuxt_grp = axM+bxM*kappaM  
    logmuxt_grp = np.repeat(logmuxt_grp[:,:,np.newaxis],nb_regions,axis=2)
    #log-likelihood      
    exp_logmuxt = np.exp(logmuxt_grp)    
    logDxtgFact = gammaln(Dxtg + 1)
    lnL         = np.sum(Dxtg * logmuxt_grp - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)          
    #dof's and numbers of records
    nb_obs  = Dxtg.size 
    dofs    = len(ax) + len(bx) + len(kappa) 
    AIC     = 2*dofs - 2*lnL    
    BIC     = dofs*np.log(nb_obs)  - 2*lnL
    #dataframe with statistics of goodness of fit
    Fit_stat = [[nb_obs,'NA','NA',dofs,np.round(lnL,2),np.round(AIC,2),np.round(BIC,2)] ]
    #We print the file
    Fit_stat         = pd.DataFrame(Fit_stat)
    Fit_stat.columns = ["N","m","degree","dofs","lnL","AIC","BIC"]    
    #we return ax, bx, kappa and stats    
    return ax, bx , kappa , Fit_stat