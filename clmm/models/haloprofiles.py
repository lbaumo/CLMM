import colossus.cosmology.cosmology as Cosmology
from colossus.utils import constants
import colossus.halo.profile_dk14 as profile_dk14
import colossus.halo as Halo
import colossus.halo.concentration as hc


import numpy as np
from scipy import integrate
from abc import ABCMeta


import numpy as np
from clmm.models import Model
import pyccl as ccl


RHO_CRITICAL = ccl.physical_constants.RHO_CRITICAL
PROFILE_PRECISION = 1e-5


class Profile(Model):
    """ Class holding halo profiles

    Class to hold halo profiles and compute DeltaSigma with them.
    Defaults to NFW but in future, overriding possible with profiles
    from CCL

    All length scales in hinv Mpc
    """
    
    def __init__(self, z_lens, mass_def, cosmology, profilename, profileparams):
        self.z_lens = z_lens
        self.mass_def = mass_def
        self.cosmology = cosmology

        if profilename == "nfw":
            self.M = profileparams['M']
            self.c = profileparams['c']

            mass_def_delta = int(self.mass_def[:-1])
            r_from_m = (0.75*self.M/(mass_def_delta*RHO_CRITICAL*np.pi))**(1./3.)
            self.rs = r_from_m/self.c
            self.delta_c = (mass_def_delta*self.c**3/3.)/(np.log(1. + self.c) - self.c/(1. + self.c))
            # self.rho_mdef = (Halo.mass_so.densityThreshold(self.zL, self.mdef) * 1E9 *(self.cosmo.h)**2.)/self.Delta

        else:
            raise ValueError("{} profile is not defined.".format(profilename))

    def profile3d():
        pass

    def profile2d():
        pass

    def sigma_r(self, rlist):
        h = self.cosmology.cosmo.params.h
        term1 = -1.0*np.ones(len(r))

        for i in range(len(rlist)):
            x = rlist[i]/self.rs
            if x < 1.0 - PROFILE_PRECISION:
                term1[i] = (1./(x*x - 1.0))*(1. - 2.*np.arctanh(np.sqrt((1. - x)/(1. + x)))/np.sqrt(1. - x*x))
            elif abs(x - 1.0) < self.esp:
                term1[i] = 1./3.
            elif x > 1.0 + self.esp:
                term1[i] = (1./(x*x - 1.))*(1. - 2.*np.arctan(np.sqrt((x - 1.)/(1. + x)))/np.sqrt(x*x - 1.))




        const = 2.*(self.rs)*self.charOverdensity()*(self.rho_mdef)
        #[Sigma] = M_dot / Mpc^2 from M{dot} / h / Mpc^2
        return (expSig * const)/self.cosmo.h 


    def sigma_mean(self, rlist):
        x = rlist/self.rs
        
        const = 4.*self.rs*self.charOverdensity()*self.rho_mdef/self.cosmo.h
        if type(x) is np.ndarray:
            fnfw = np.piecewise(x,[x<1., x==1., x>1.], \
                        [lambda x:(2./np.sqrt(1.-x*x)*np.arctanh(np.sqrt((1.-x)/(1.+x)))+np.log(x/2.))/(x*x), \
                         lambda x:(1.+np.log(0.5)), \
                         lambda x:(2./np.sqrt(x*x-1.)*np.arctan2(np.sqrt(x-1.),np.sqrt(1.+x))+np.log(x/2.))/(x*x)])
            return const*fnfw
        pass

    def sigma_crit():
        pass

    def delta_sigma():
        pass






class profile(object):
    

    '''
    ############################################################################
                          Critical Surface Mass Density
    ############################################################################
    '''
    
    def Sc(self, zS):
        # [D] = Mpc/h
        Dl = self.cosmo.angularDiameterDistance(self.zL)
        Ds = self.cosmo.angularDiameterDistance(zS) 
        Dls = Ds - (1. + self.zL) * Dl /(1.+zS) #assuming flat cosmology
        ret = (self.v_c)**2. / (4.0 * np.pi * self.G) * Ds / ( Dl * Dls)
        #ret = ret/((1.+self.zL)**2.) # from the use of comoving scale
        #[Sc] = M_dot / Mpc^2 from M{dot} h / Mpc^2
        return ret * self.cosmo.h


    '''
    ############################################################################
                                  rho & Sigma
    ############################################################################
    '''
    
    def rho(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            rho = self.nfwrho(r)
            
        return rho
    
    def Sigma(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            Sigma = self.nfwSigma(r)
            
        return Sigma
    
    
    '''
    ############################################################################
                                  SigmaMean
    ############################################################################
    '''
    
    def SigmaMean(self,r):
        #[r] = Mpc/h
        
        if self.profile == 'nfw':
            SigmaMean = self.nfwSigmaMean(r)
        #set first bin of SigmaMean[0] to Sigma[0]
        
            
        #SigmaMean[0] = self.Sigma(r[0])
        return SigmaMean
    
    def deltaSigma(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            dSig = self.nfwDeltaSigma(r)
        return dSig
        

'''
############################################################################
                                  NFW
############################################################################
'''
class nfwProfile(profile):
    ##### We're going to swap out ``profile'' with ``Profile1D''
    def __init__(self, M , c , zL, mdef, chooseCosmology, esp = None):
        profile.__init__(self, zL, mdef, chooseCosmology)
        
        #self.parameters = parameters
        self.M_mdef = M #M200 input in M_dot/h
        
        self.c = c
        self.zL = zL
        self.mdef = mdef
        self.chooseCosmology = chooseCosmology
        self.G, self.v_c, self.H2, self.cosmo = self.calcConstants()
        self.r_mdef = Halo.mass_so.M_to_R(self.M_mdef, self.zL, self.mdef)/1E3 #Mpc/h
        self.Delta = int(mdef[:-1])
        #[rho_mdef] = M_dot Mpc^3 from M_{\odot}h^2/kpc^3
        self.rho_mdef = (Halo.mass_so.densityThreshold(self.zL, self.mdef) * 1E9 *(self.cosmo.h)**2.)/self.Delta
        self.rs = self.r_mdef/self.c #Mpc/h
        self.profile = 'nfw'
        if esp == None:
            self.esp = 1E-5
        else:
            self.esp = esp
        return
        
    '''
    ############################################################################
                               Analytic NFW profile
    ############################################################################
    '''
    def charOverdensity(self):
        Delta = int(self.mdef[:-1])
        sigma_c = (Delta/3.)*(self.c**3.)/(np.log(1. + self.c) - self.c/(1. + self.c))
        return sigma_c #unitless
    
    def nfwrho(self, R):
        #R in Mpc/h
        #[sigma_c] = unitless
        #[rho_mdef] = M_dot / Mpc^3
        
        const =  self.rho_mdef * self.charOverdensity() 
        rhoForm = 1./( (R/self.rs) * (1. + R/self.rs)**2.)
        return (const * rhoForm)
        
    def nfwSigma(self, r):
        #[r] = Mpc/h
        
        #[rs] = Mpc/h        
        rs = self.rs
        expSig = np.empty(len(r))
        for i in range(len(r)):
            if r[i]/self.rs < 1.0 - self.esp:
                expSig[i] = (1./((r[i]/rs)**2. - 1.))*(1. - 2.*np.arctanh(np.sqrt(\
                (1. - (r[i]/rs))/(1. + (r[i]/rs))))/np.sqrt(1. - (r[i]/rs)**2.))
            #if r[i]/rs == 1.0:
            #    expSig[i] = 1./3.
            if r[i]/rs >= 1.0 - self.esp and r[i]/rs <= 1.0 + self.esp:
                expSig[i] = 1./3.
            if r[i]/self.rs > 1.0 + self.esp:
                expSig[i] = (1./((r[i]/rs)**2. - 1.))*(1. - 2.*np.arctan(np.sqrt(\
                ((r[i]/rs) - 1.)/(1. + (r[i]/rs))))/np.sqrt((r[i]/rs)**2. - 1.))
        const = 2.*(self.rs)*self.charOverdensity()*(self.rho_mdef)
        #[Sigma] = M_dot / Mpc^2 from M{dot} / h / Mpc^2
        return (expSig * const)/self.cosmo.h 
    
    def nfwSigmaMean(self, r):
        #[r] = Mpc/h
        x = r/self.rs
        const = 4.*self.rs*self.charOverdensity()*self.rho_mdef/self.cosmo.h
        if type(x) is np.ndarray:
            #print x
            fnfw = np.piecewise(x,[x<1., x==1., x>1.], \
                        [lambda x:(2./np.sqrt(1.-x*x)*np.arctanh(np.sqrt((1.-x)/(1.+x)))+np.log(x/2.))/(x*x), \
                         lambda x:(1.+np.log(0.5)), \
                         lambda x:(2./np.sqrt(x*x-1.)*np.arctan2(np.sqrt(x-1.),np.sqrt(1.+x))+np.log(x/2.))/(x*x)])
            return const*fnfw
    
        else:
            if x<1:
                return const*(2./np.sqrt(1.-x*x)*np.arctanh(np.sqrt((1.-x)/(1.+x)))+np.log(x/2.))/(x*x)
            elif x==1:
                return const*(1.+np.log(0.5))
            else:
                return const*(2./np.sqrt(x*x-1.)*np.arctan2(np.sqrt(x-1.),np.sqrt(1.+x))+np.log(x/2.))/(x*x)
        
        
    
    def nfwDeltaSigma(self, r):
        #[r] = Mpc/h
        rs = self.rs
        
        x = r/rs
        expG = np.empty(len(x))
        for i in range(len(x)):
            if x[i] < 1.0 - self.esp:
                expG[i] = 8.0*np.arctanh(np.sqrt((1.0-(x[i]))/(1.0+(x[i]))))/(((x[i])**2.0)*np.sqrt(1.0-(x[i])**2.0))\
                + (4.0/((x[i])**2.0))*np.log((x[i])/2.0) \
                - 2.0/((x[i])**2.0 - 1.0)\
                + 4.0*np.arctanh(np.sqrt((1.0-(x[i]))/(1.0+(x[i]))))/(((x[i])**2.0 - 1.0)*np.sqrt(1.0-(x[i])**2.0))
            if x[i] > 1.0 - self.esp and x[i] < 1.0 + self.esp:
                expG[i] = (10.0 / 3.0) + (4.0 * np.log(1.0/2.0))
            if x[i] > 1.0 + self.esp:
                expG[i] = 8.0*np.arctan(np.sqrt(((x[i])-1.0)/(1.0+(x[i]))))/(((x[i])**2.0)*np.sqrt((x[i])**2.0 - 1.0))\
                + (4.0/((x[i])**2.0))*np.log((x[i])/2.0) \
                - 2.0/((x[i])**2.0 - 1.0)\
                + 4.0*np.arctan(np.sqrt(((x[i])-1.0)/(1.0+(x[i]))))/(((x[i])**2.0 - 1.0)**(3.0/2.0))
        
        
        #[rho_mdef] = M_dot/Mpc^3 
        #[charOverdensity] = unitless
        charOverdensity = self.charOverdensity()
        Const = (rs) * (self.rho_mdef) * (charOverdensity)
        # [Const] = 1 from h^-1
        Const = Const/(self.cosmo.h)
        return Const*expG
    
        
        
    

