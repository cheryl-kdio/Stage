import numpy as np
import warnings
from scipy.optimize import minimize


class UnivariateHawkesMLE:

    """
    Hawkes univarié exponentiel classique.

    lambda(t)=mu+sum_{t_k<t} alpha*beta*exp(-beta*(t-t_k))

    Paramètres estimés :
        mu    > 0
        alpha >= 0
        beta  > 0

    Compensateur :
        int_0^T lambda(u)du
        =
        mu*T+alpha*sum_k(1-exp(-beta*(T-t_k)))

    Condition de stabilité :
        alpha < 1
    """

    def __init__(self,beta_init=1.0,max_iter=3000,tol=1e-8,
                 min_baseline=1e-12,min_decay=1e-8,
                 alpha_upper=None,beta_upper=None,
                 alpha_l2=0.0,beta_l2=0.0,
                 n_starts=1,random_state=None):
        self.beta_init=float(beta_init)
        self.max_iter=int(max_iter)
        self.tol=float(tol)
        self.min_baseline=float(min_baseline)
        self.min_decay=float(min_decay)
        self.alpha_upper=alpha_upper
        self.beta_upper=beta_upper
        self.alpha_l2=float(alpha_l2)
        self.beta_l2=float(beta_l2)
        self.n_starts=int(n_starts)
        self.random_state=random_state
        if self.beta_init<=0:
            raise ValueError("beta_init doit être strictement positif.")

    @staticmethod
    def _prepare_one_realization(times):
        times=np.asarray(times,dtype=float).ravel()
        if np.any(~np.isfinite(times)):
            raise ValueError("times doit contenir des valeurs finies.")
        return np.sort(times)

    def _prepare_realizations(self,events,end_times=None):
        if isinstance(events,np.ndarray):
            realizations=[self._prepare_one_realization(events)]
        elif isinstance(events,(list,tuple)):
            if len(events)==0:
                raise ValueError("events ne peut pas être vide.")
            if all(np.ndim(x)==0 for x in events):
                realizations=[self._prepare_one_realization(events)]
            else:
                realizations=[self._prepare_one_realization(ev) for ev in events]
        else:
            realizations=[self._prepare_one_realization(events)]

        if end_times is None:
            Ts=[]
            for t in realizations:
                if len(t)==0:
                    raise ValueError("end_times est requis si une réalisation est vide.")
                Ts.append(float(t[-1]))
            warnings.warn(
                "end_times non fourni : utilisation du dernier timestamp. "
                "Pour une MLE correcte, fournissez l'horizon réel d'observation.",
                RuntimeWarning
            )
        elif np.ndim(end_times)==0:
            Ts=[float(end_times)]*len(realizations)
        else:
            Ts=[float(x) for x in np.asarray(end_times,dtype=float).ravel()]
            if len(Ts)!=len(realizations):
                raise ValueError("end_times doit être scalaire ou de longueur n_realizations.")

        for idx,(t,T) in enumerate(zip(realizations,Ts)):
            if T<=0 or not np.isfinite(T):
                raise ValueError("Chaque end_time doit être strictement positif et fini.")
            if np.any(t<0):
                raise ValueError(f"Réalisation {idx}: timestamps négatifs.")
            if np.any(t>T):
                raise ValueError(f"Réalisation {idx}: timestamps au-delà de end_time.")

        return realizations,np.asarray(Ts,dtype=float)

    @staticmethod
    def _unpack(theta):
        mu,alpha,beta=theta
        return float(mu),float(alpha),float(beta)

    @staticmethod
    def _kernel_integral_and_grad(times,T,beta):
        if len(times)==0:
            return 0.0,0.0
        rem=T-times
        e=np.exp(-beta*rem)
        S=np.sum(1.0-e)
        S_beta=np.sum(rem*e)
        return float(S),float(S_beta)

    def _nll_grad_one(self,theta,times,T):
        mu,alpha,beta=self._unpack(theta)
        if mu<=0 or alpha<0 or beta<=0:
            return np.inf,np.zeros_like(theta)

        ll=0.0
        grad_mu=grad_alpha=grad_beta=0.0
        r=q=0.0
        last_t=0.0
        n,k=len(times),0

        while k<n:
            t=times[k]
            dt=t-last_t
            if dt<-1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")

            if dt>0:
                e=np.exp(-beta*dt)
                q=e*(q-dt*r)
                r=e*r

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            count=k2-k
            lam=mu+alpha*r
            if lam<=0 or not np.isfinite(lam):
                return np.inf,np.zeros_like(theta)

            inv_lam=1.0/lam
            ll+=count*np.log(lam)
            grad_mu+=count*inv_lam
            grad_alpha+=count*r*inv_lam
            grad_beta+=count*alpha*q*inv_lam

            r+=beta*count
            q+=count

            last_t=t
            k=k2

        S,S_beta=self._kernel_integral_and_grad(times,T,beta)

        ll-=mu*T+alpha*S
        grad_mu-=T
        grad_alpha-=S
        grad_beta-=alpha*S_beta

        nll=-ll
        grad=np.array([-grad_mu,-grad_alpha,-grad_beta],dtype=float)

        if self.alpha_l2>0:
            nll+=0.5*self.alpha_l2*alpha*alpha
            grad[1]+=self.alpha_l2*alpha
        if self.beta_l2>0:
            nll+=0.5*self.beta_l2*beta*beta
            grad[2]+=self.beta_l2*beta

        return float(nll),grad

    def _nll_grad_all(self,theta,realizations,end_times):
        nll_total=0.0
        grad_total=np.zeros_like(theta,dtype=float)

        for times,T in zip(realizations,end_times):
            nll,grad=self._nll_grad_one(theta,times,float(T))
            if not np.isfinite(nll):
                return np.inf,np.zeros_like(theta)
            nll_total+=nll
            grad_total+=grad

        return float(nll_total),grad_total

    def _initial_theta(self,realizations,end_times):
        total_events=sum(len(x) for x in realizations)
        total_T=float(np.sum(end_times))
        mu0=max(0.5*total_events/max(total_T,1e-12),self.min_baseline*10)
        alpha0=0.05
        beta0=max(self.beta_init,self.min_decay*10)

        if self.beta_upper is not None:
            beta0=min(beta0,self.beta_upper*0.9)

        return np.array([mu0,alpha0,beta0],dtype=float)

    def fit(self,events,end_times=None,x0=None):
        realizations,end_times=self._prepare_realizations(events,end_times)

        if x0 is None:
            theta0=self._initial_theta(realizations,end_times)
        else:
            theta0=np.asarray(x0,dtype=float).ravel()
            if theta0.size!=3:
                raise ValueError("x0 doit avoir une longueur 3 : [mu,alpha,beta].")

        bounds=[
            (self.min_baseline,None),
            (0.0,self.alpha_upper),
            (self.min_decay,self.beta_upper)
        ]

        rng=np.random.default_rng(self.random_state)
        best_result,best_fun=None,np.inf

        for start in range(max(1,self.n_starts)):
            if start==0:
                start_theta=theta0.copy()
            else:
                mu0,alpha0,beta0=theta0
                start_theta=np.array([
                    mu0*rng.lognormal(0.0,0.4),
                    alpha0*rng.lognormal(0.0,0.7),
                    beta0*rng.lognormal(0.0,0.7)
                ],dtype=float)

            for idx,(lo,hi) in enumerate(bounds):
                if lo is not None and start_theta[idx]<lo:
                    start_theta[idx]=lo*10.0 if lo>0 else lo
                if hi is not None and start_theta[idx]>hi:
                    start_theta[idx]=hi*0.9 if hi>0 else hi

            result=minimize(
                fun=lambda th:self._nll_grad_all(th,realizations,end_times),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter":self.max_iter,"ftol":self.tol}
            )

            if result.fun<best_fun:
                best_fun=float(result.fun)
                best_result=result

        self.result_=best_result
        self.success_=bool(best_result.success)
        self.message_=best_result.message
        self.n_iter_=best_result.nit
        self.events_=realizations
        self.end_times_=end_times
        self.baseline_,self.alpha_,self.beta_=self._unpack(best_result.x)
        self.log_likelihood_=-float(best_result.fun)
        self.branching_ratio_=float(self.alpha_)
        self.is_stable_=bool(self.branching_ratio_<1.0)

        return self

    def score(self,events=None,end_times=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations=self.events_
            end_times=self.end_times_
        else:
            realizations,end_times=self._prepare_realizations(events,end_times)

        theta=np.array([self.baseline_,self.alpha_,self.beta_],dtype=float)
        nll,_=self._nll_grad_all(theta,realizations,end_times)
        return -float(nll)

    def intensity_at_events(self,events=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant intensity_at_events().")

        times=self.events_[0] if events is None else self._prepare_one_realization(events)
        mu,alpha,beta=self.baseline_,self.alpha_,self.beta_
        intensities=np.zeros(len(times),dtype=float)
        r=0.0
        last_t=0.0
        n,k=len(times),0

        while k<n:
            t=times[k]
            dt=t-last_t
            if dt>0:
                r*=np.exp(-beta*dt)

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            intensities[k:k2]=mu+alpha*r
            r+=beta*(k2-k)
            last_t=t
            k=k2

        return intensities

    def get_params(self):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline":self.baseline_,
            "alpha":self.alpha_,
            "beta":self.beta_,
            "branching_ratio":self.branching_ratio_,
            "is_stable":self.is_stable_,
            "log_likelihood":self.log_likelihood_,
            "success":self.success_,
            "message":self.message_,
            "n_iter":self.n_iter_
        }
    

class UnivariateMarkedHawkesMLE2:
    """
    Hawkes univarié avec decay modulé par z_k.

    lambda(t)=mu+sum_{t_k<t} alpha*beta*exp(-beta*(t-t_k)/z_k)

    Paramètres estimés :
        mu    > 0
        alpha >= 0
        beta  > 0

    Les z_k sont observés et doivent être strictement positifs.

    Compensateur :
        int_0^T lambda(u)du
        =
        mu*T+alpha*sum_k z_k*(1-exp(-beta*(T-t_k)/z_k))

    Attention :
        si les z_k sont arbitraires, le calcul exact des intensités
        aux événements est en O(n^2).
    """

    def __init__(self,beta_init=1.0,max_iter=3000,tol=1e-8,
                 min_baseline=1e-12,min_decay=1e-8,
                 alpha_upper=None,beta_upper=None,
                 alpha_l2=0.0,beta_l2=0.0,
                 n_starts=1,random_state=None):
        self.beta_init=float(beta_init)
        self.max_iter=int(max_iter)
        self.tol=float(tol)
        self.min_baseline=float(min_baseline)
        self.min_decay=float(min_decay)
        self.alpha_upper=alpha_upper
        self.beta_upper=beta_upper
        self.alpha_l2=float(alpha_l2)
        self.beta_l2=float(beta_l2)
        self.n_starts=int(n_starts)
        self.random_state=random_state
        if self.beta_init<=0:
            raise ValueError("beta_init doit être strictement positif.")

    @staticmethod
    def _prepare_one_realization(times,z=None):
        times=np.asarray(times,dtype=float).ravel()
        z=np.ones_like(times,dtype=float) if z is None else np.asarray(z,dtype=float).ravel()

        if times.shape!=z.shape:
            raise ValueError("times et z doivent avoir la même longueur.")
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(z)):
            raise ValueError("times et z doivent contenir des valeurs finies.")
        if np.any(z<=0):
            raise ValueError("Tous les z_k doivent être strictement positifs.")

        order=np.argsort(times)
        return times[order],z[order]

    def _prepare_realizations(self,events,z=None,end_times=None):
        if isinstance(events,np.ndarray):
            t,zz=self._prepare_one_realization(events,z)
            realizations,z_realizations=[t],[zz]
        elif isinstance(events,(list,tuple)):
            if len(events)==0:
                raise ValueError("events ne peut pas être vide.")
            if all(np.ndim(x)==0 for x in events):
                t,zz=self._prepare_one_realization(events,z)
                realizations,z_realizations=[t],[zz]
            else:
                realizations,z_realizations=[],[]
                if z is None:
                    z_iter=[None]*len(events)
                else:
                    if len(z)!=len(events):
                        raise ValueError("Pour plusieurs réalisations, z doit avoir la même longueur que events.")
                    z_iter=z
                for ev,zk in zip(events,z_iter):
                    t,zz=self._prepare_one_realization(ev,zk)
                    realizations.append(t)
                    z_realizations.append(zz)
        else:
            t,zz=self._prepare_one_realization(events,z)
            realizations,z_realizations=[t],[zz]

        if end_times is None:
            Ts=[]
            for t in realizations:
                if len(t)==0:
                    raise ValueError("end_times est requis si une réalisation est vide.")
                Ts.append(float(t[-1]))
            warnings.warn(
                "end_times non fourni : utilisation du dernier timestamp. "
                "Pour une MLE correcte, fournissez l'horizon réel d'observation.",
                RuntimeWarning
            )
        elif np.ndim(end_times)==0:
            Ts=[float(end_times)]*len(realizations)
        else:
            Ts=[float(x) for x in np.asarray(end_times,dtype=float).ravel()]
            if len(Ts)!=len(realizations):
                raise ValueError("end_times doit être scalaire ou de longueur n_realizations.")

        for idx,(t,T) in enumerate(zip(realizations,Ts)):
            if T<=0 or not np.isfinite(T):
                raise ValueError("Chaque end_time doit être strictement positif et fini.")
            if np.any(t<0):
                raise ValueError(f"Réalisation {idx}: timestamps négatifs.")
            if np.any(t>T):
                raise ValueError(f"Réalisation {idx}: timestamps au-delà de end_time.")

        return realizations,z_realizations,np.asarray(Ts,dtype=float)

    @staticmethod
    def _unpack(theta):
        mu,alpha,beta=theta
        return float(mu),float(alpha),float(beta)

    @staticmethod
    def _kernel_integral_and_grad(times,z,T,beta):
        if len(times)==0:
            return 0.0,0.0

        rem=T-times
        e=np.exp(-beta*rem/z)

        S=np.sum(z*(1.0-e))
        S_beta=np.sum(rem*e)

        return float(S),float(S_beta)

    def _nll_grad_one(self,theta,times,z,T):
        mu,alpha,beta=self._unpack(theta)
        if mu<=0 or alpha<0 or beta<=0:
            return np.inf,np.zeros_like(theta)

        ll=0.0
        grad_mu=grad_alpha=grad_beta=0.0
        n,k=len(times),0

        while k<n:
            t=times[k]

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            count=k2-k

            if k==0:
                A=0.0
                B=0.0
            else:
                dt=t-times[:k]
                past_z=z[:k]
                e=np.exp(-beta*dt/past_z)
                A=np.sum(e)
                B=np.sum((dt/past_z)*e)

            lam=mu+alpha*beta*A
            if lam<=0 or not np.isfinite(lam):
                return np.inf,np.zeros_like(theta)

            inv_lam=1.0/lam
            ll+=count*np.log(lam)

            grad_mu+=count*inv_lam
            grad_alpha+=count*beta*A*inv_lam
            grad_beta+=count*alpha*(A-beta*B)*inv_lam

            k=k2

        S,S_beta=self._kernel_integral_and_grad(times,z,T,beta)

        ll-=mu*T+alpha*S
        grad_mu-=T
        grad_alpha-=S
        grad_beta-=alpha*S_beta

        nll=-ll
        grad=np.array([-grad_mu,-grad_alpha,-grad_beta],dtype=float)

        if self.alpha_l2>0:
            nll+=0.5*self.alpha_l2*alpha*alpha
            grad[1]+=self.alpha_l2*alpha
        if self.beta_l2>0:
            nll+=0.5*self.beta_l2*beta*beta
            grad[2]+=self.beta_l2*beta

        return float(nll),grad

    def _nll_grad_all(self,theta,realizations,z_realizations,end_times):
        nll_total=0.0
        grad_total=np.zeros_like(theta,dtype=float)

        for times,z,T in zip(realizations,z_realizations,end_times):
            nll,grad=self._nll_grad_one(theta,times,z,float(T))
            if not np.isfinite(nll):
                return np.inf,np.zeros_like(theta)
            nll_total+=nll
            grad_total+=grad

        return float(nll_total),grad_total

    def _initial_theta(self,realizations,end_times):
        total_events=sum(len(x) for x in realizations)
        total_T=float(np.sum(end_times))
        mu0=max(0.5*total_events/max(total_T,1e-12),self.min_baseline*10)
        alpha0=0.05
        beta0=max(self.beta_init,self.min_decay*10)

        if self.beta_upper is not None:
            beta0=min(beta0,self.beta_upper*0.9)

        return np.array([mu0,alpha0,beta0],dtype=float)

    def fit(self,events,z=None,end_times=None,x0=None):
        realizations,z_realizations,end_times=self._prepare_realizations(events,z,end_times)

        if x0 is None:
            theta0=self._initial_theta(realizations,end_times)
        else:
            theta0=np.asarray(x0,dtype=float).ravel()
            if theta0.size!=3:
                raise ValueError("x0 doit avoir une longueur 3 : [mu,alpha,beta].")

        bounds=[
            (self.min_baseline,None),
            (0.0,self.alpha_upper),
            (self.min_decay,self.beta_upper)
        ]

        rng=np.random.default_rng(self.random_state)
        best_result,best_fun=None,np.inf

        for start in range(max(1,self.n_starts)):
            if start==0:
                start_theta=theta0.copy()
            else:
                mu0,alpha0,beta0=theta0
                start_theta=np.array([
                    mu0*rng.lognormal(0.0,0.4),
                    alpha0*rng.lognormal(0.0,0.7),
                    beta0*rng.lognormal(0.0,0.7)
                ],dtype=float)

            for idx,(lo,hi) in enumerate(bounds):
                if lo is not None and start_theta[idx]<lo:
                    start_theta[idx]=lo*10.0 if lo>0 else lo
                if hi is not None and start_theta[idx]>hi:
                    start_theta[idx]=hi*0.9 if hi>0 else hi

            result=minimize(
                fun=lambda th:self._nll_grad_all(th,realizations,z_realizations,end_times),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter":self.max_iter,"ftol":self.tol}
            )

            if result.fun<best_fun:
                best_fun=float(result.fun)
                best_result=result

        self.result_=best_result
        self.success_=bool(best_result.success)
        self.message_=best_result.message
        self.n_iter_=best_result.nit
        self.events_=realizations
        self.z_=z_realizations
        self.end_times_=end_times
        self.baseline_,self.alpha_,self.beta_=self._unpack(best_result.x)
        self.log_likelihood_=-float(best_result.fun)

        all_z=np.concatenate(z_realizations) if sum(len(x) for x in z_realizations)>0 else np.array([])
        self.mean_z_=float(np.mean(all_z)) if len(all_z)>0 else 1.0

        # Ici l'intégrale totale du noyau associé à un événement de mark z_k vaut alpha*z_k.
        # Une notion empirique de branching moyen est donc alpha*mean(z_k).
        self.branching_ratio_empirical_=float(self.alpha_*self.mean_z_)
        self.is_stable_empirical_=bool(self.branching_ratio_empirical_<1.0)

        return self

    def score(self,events=None,z=None,end_times=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations=self.events_
            z_realizations=self.z_
            end_times=self.end_times_
        else:
            realizations,z_realizations,end_times=self._prepare_realizations(events,z,end_times)

        theta=np.array([self.baseline_,self.alpha_,self.beta_],dtype=float)
        nll,_=self._nll_grad_all(theta,realizations,z_realizations,end_times)

        return -float(nll)

    def intensity_at_events(self,events=None,z=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant intensity_at_events().")

        if events is None:
            times=self.events_[0]
            z=self.z_[0]
        else:
            times,z=self._prepare_one_realization(events,z)

        mu,alpha,beta=self.baseline_,self.alpha_,self.beta_
        intensities=np.zeros(len(times),dtype=float)
        n,k=len(times),0

        while k<n:
            t=times[k]

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            if k==0:
                A=0.0
            else:
                dt=t-times[:k]
                A=np.sum(np.exp(-beta*dt/z[:k]))

            intensities[k:k2]=mu+alpha*beta*A
            k=k2

        return intensities

    def get_params(self):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline":self.baseline_,
            "alpha":self.alpha_,
            "beta":self.beta_,
            "mean_z":self.mean_z_,
            "branching_ratio_empirical":self.branching_ratio_empirical_,
            "is_stable_empirical":self.is_stable_empirical_,
            "log_likelihood":self.log_likelihood_,
            "success":self.success_,
            "message":self.message_,
            "n_iter":self.n_iter_
        }
    
class UnivariateMarkedHawkesMLE1:
    """
    Hawkes univarié exponentiel avec mark événementiel.

    lambda(t)=mu+sum_{t_k<t} alpha*exp(eta*z_k)*beta*exp(-beta*(t-t_k))

    Ici :
        z_k = standardisation du mark brut.
        beta est toujours estimé.
    """

    def __init__(self,beta_init=1.0,max_iter=3000,tol=1e-8,min_baseline=1e-12,
                 min_decay=1e-8,alpha_upper=None,beta_upper=None,eta_bounds=(-5.0,5.0),
                 alpha_l2=0.0,beta_l2=0.0,eta_l2=0.0,n_starts=1,random_state=None):
        self.beta_init=float(beta_init)
        self.max_iter=int(max_iter)
        self.tol=float(tol)
        self.min_baseline=float(min_baseline)
        self.min_decay=float(min_decay)
        self.alpha_upper=alpha_upper
        self.beta_upper=beta_upper
        self.eta_bounds=eta_bounds
        self.alpha_l2=float(alpha_l2)
        self.beta_l2=float(beta_l2)
        self.eta_l2=float(eta_l2)
        self.n_starts=int(n_starts)
        self.random_state=random_state
        if self.beta_init<=0:
            raise ValueError("beta_init doit être strictement positif.")

    @staticmethod
    def _prepare_one_realization(times,marks=None):
        times=np.asarray(times,dtype=float).ravel()
        marks=np.ones_like(times,dtype=float) if marks is None else np.asarray(marks,dtype=float).ravel()
        if times.shape!=marks.shape:
            raise ValueError("times et marks doivent avoir la même longueur.")
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(marks)):
            raise ValueError("times et marks doivent contenir des valeurs finies.")
        order=np.argsort(times)
        return times[order],marks[order]

    def _prepare_realizations(self,events,marks=None,end_times=None):
        if isinstance(events,np.ndarray):
            t,m=self._prepare_one_realization(events,marks)
            realizations,marks_realizations=[t],[m]
        elif isinstance(events,(list,tuple)):
            if len(events)==0:
                raise ValueError("events ne peut pas être vide.")
            if all(np.ndim(x)==0 for x in events):
                t,m=self._prepare_one_realization(events,marks)
                realizations,marks_realizations=[t],[m]
            else:
                realizations,marks_realizations=[],[]
                if marks is None:
                    marks_iter=[None]*len(events)
                else:
                    if len(marks)!=len(events):
                        raise ValueError("Pour plusieurs réalisations, marks doit avoir la même longueur que events.")
                    marks_iter=marks
                for ev,mk in zip(events,marks_iter):
                    t,m=self._prepare_one_realization(ev,mk)
                    realizations.append(t)
                    marks_realizations.append(m)
        else:
            t,m=self._prepare_one_realization(events,marks)
            realizations,marks_realizations=[t],[m]

        if end_times is None:
            Ts=[]
            for t in realizations:
                if len(t)==0:
                    raise ValueError("end_times est requis si une réalisation ne contient aucun événement.")
                Ts.append(float(t[-1]))
            warnings.warn(
                "end_times non fourni : utilisation du dernier timestamp de chaque réalisation. "
                "Pour une MLE correcte, fournissez l'horizon réel d'observation.",
                RuntimeWarning
            )
        elif np.ndim(end_times)==0:
            Ts=[float(end_times)]*len(realizations)
        else:
            Ts=[float(x) for x in np.asarray(end_times,dtype=float).ravel()]
            if len(Ts)!=len(realizations):
                raise ValueError("end_times doit être un scalaire ou un array de longueur n_realizations.")

        for idx,(t,T) in enumerate(zip(realizations,Ts)):
            if T<=0 or not np.isfinite(T):
                raise ValueError("Chaque end_time doit être strictement positif et fini.")
            if np.any(t<0):
                raise ValueError(f"Réalisation {idx}: timestamps négatifs.")
            if np.any(t>T):
                raise ValueError(f"Réalisation {idx}: certains timestamps dépassent end_time.")
        return realizations,marks_realizations,np.asarray(Ts,dtype=float)

    def _fit_mark_standardization(self,marks_realizations):
        raw_all=[np.asarray(marks,dtype=float).ravel() for marks in marks_realizations]
        if len(raw_all)==0:
            raise ValueError("Aucun mark disponible.")
        concatenated=np.concatenate(raw_all) if sum(len(x) for x in raw_all)>0 else np.array([])
        if len(concatenated)==0:
            mean,std=0.0,1.0
        else:
            mean,std=float(np.mean(concatenated)),float(np.std(concatenated))
            if std<=1e-12:
                std=1.0
        z_realizations=[(raw-mean)/std for raw in raw_all]
        return z_realizations,{"mean":mean,"std":std}

    def _transform_marks_with_stats(self,marks_realizations,stats):
        return [(np.asarray(marks,dtype=float).ravel()-stats["mean"])/stats["std"] for marks in marks_realizations]

    @staticmethod
    def _unpack(theta):
        mu,alpha,beta,eta=theta
        return float(mu),float(alpha),float(beta),float(eta)

    @staticmethod
    def _kernel_integral_and_grads(times,z,T,beta,eta):
        if len(times)==0:
            return 0.0,0.0,0.0
        rem=T-times
        w=np.exp(eta*z)
        e=np.exp(-beta*rem)
        S=np.sum(w*(1.0-e))
        S_eta=np.sum(w*z*(1.0-e))
        S_beta=np.sum(w*rem*e)
        return float(S),float(S_eta),float(S_beta)

    def _nll_grad_one(self,theta,times,z,T):
        mu,alpha,beta,eta=self._unpack(theta)
        if mu<=0 or alpha<0 or beta<=0:
            return np.inf,np.zeros_like(theta)

        ll=0.0
        grad_mu=grad_alpha=grad_beta=grad_eta=0.0
        r=q=u=0.0
        last_t=0.0
        n,k=len(times),0

        while k<n:
            t=times[k]
            dt=t-last_t
            if dt<-1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")
            if dt>0:
                e=np.exp(-beta*dt)
                q=e*(q-dt*r)
                r=e*r
                u=e*u

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            z_group=z[k:k2]
            count=k2-k
            lam=mu+alpha*r
            if lam<=0 or not np.isfinite(lam):
                return np.inf,np.zeros_like(theta)

            ll+=count*np.log(lam)
            inv_lam=1.0/lam
            grad_mu+=count*inv_lam
            grad_alpha+=count*r*inv_lam
            grad_beta+=count*alpha*q*inv_lam
            grad_eta+=count*alpha*u*inv_lam

            w_group=np.exp(eta*z_group)
            w_sum=np.sum(w_group)
            wz_sum=np.sum(w_group*z_group)

            r+=beta*w_sum
            q+=w_sum
            u+=beta*wz_sum

            last_t=t
            k=k2

        S,S_eta,S_beta=self._kernel_integral_and_grads(times,z,T,beta,eta)

        ll-=mu*T+alpha*S
        grad_mu-=T
        grad_alpha-=S
        grad_beta-=alpha*S_beta
        grad_eta-=alpha*S_eta

        nll=-ll
        grad_mu=-grad_mu
        grad_alpha=-grad_alpha
        grad_beta=-grad_beta
        grad_eta=-grad_eta

        if self.alpha_l2>0:
            nll+=0.5*self.alpha_l2*alpha*alpha
            grad_alpha+=self.alpha_l2*alpha
        if self.beta_l2>0:
            nll+=0.5*self.beta_l2*beta*beta
            grad_beta+=self.beta_l2*beta
        if self.eta_l2>0:
            nll+=0.5*self.eta_l2*eta*eta
            grad_eta+=self.eta_l2*eta

        grad=np.array([grad_mu,grad_alpha,grad_beta,grad_eta],dtype=float)
        return float(nll),grad

    def _nll_grad_all(self,theta,realizations,z_realizations,end_times):
        nll_total=0.0
        grad_total=np.zeros_like(theta,dtype=float)
        for times,z,T in zip(realizations,z_realizations,end_times):
            nll,grad=self._nll_grad_one(theta,times,z,float(T))
            if not np.isfinite(nll):
                return np.inf,np.zeros_like(theta)
            nll_total+=nll
            grad_total+=grad
        return float(nll_total),grad_total

    def _initial_theta(self,realizations,end_times):
        total_events=sum(len(x) for x in realizations)
        total_T=float(np.sum(end_times))
        mu0=max(0.5*total_events/max(total_T,1e-12),self.min_baseline*10)
        alpha0=0.05
        beta0=max(self.beta_init,self.min_decay*10)
        eta0=0.0
        if self.beta_upper is not None:
            beta0=min(beta0,self.beta_upper*0.9)
        return np.array([mu0,alpha0,beta0,eta0],dtype=float)

    def fit(self,events,marks=None,end_times=None,x0=None):
        realizations,marks_realizations,end_times=self._prepare_realizations(events,marks,end_times)
        z_realizations,mark_stats=self._fit_mark_standardization(marks_realizations)

        if x0 is None:
            theta0=self._initial_theta(realizations,end_times)
        else:
            theta0=np.asarray(x0,dtype=float).ravel()
            if theta0.size!=4:
                raise ValueError("x0 doit avoir une longueur 4.")

        bounds=[
            (self.min_baseline,None),
            (0.0,self.alpha_upper),
            (self.min_decay,self.beta_upper),
            self.eta_bounds
        ]

        rng=np.random.default_rng(self.random_state)
        best_result,best_fun=None,np.inf

        for start in range(max(1,self.n_starts)):
            if start==0:
                start_theta=theta0.copy()
            else:
                mu0,alpha0,beta0,eta0=theta0
                start_theta=np.array([
                    mu0*rng.lognormal(0.0,0.4),
                    alpha0*rng.lognormal(0.0,0.7),
                    beta0*rng.lognormal(0.0,0.7),
                    eta0+rng.normal(0.0,0.4)
                ],dtype=float)

            for idx,(lo,hi) in enumerate(bounds):
                if lo is not None and start_theta[idx]<lo:
                    start_theta[idx]=lo*10.0 if lo>0 else lo
                if hi is not None and start_theta[idx]>hi:
                    start_theta[idx]=hi*0.9 if hi>0 else hi

            result=minimize(
                fun=lambda th:self._nll_grad_all(th,realizations,z_realizations,end_times),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter":self.max_iter,"ftol":self.tol}
            )

            if result.fun<best_fun:
                best_fun=float(result.fun)
                best_result=result

        self.result_=best_result
        self.success_=bool(best_result.success)
        self.message_=best_result.message
        self.n_iter_=best_result.nit
        self.events_=realizations
        self.marks_=marks_realizations
        self.z_marks_=z_realizations
        self.end_times_=end_times
        self.mark_stats_=mark_stats
        self.baseline_,self.alpha_,self.beta_,self.eta_=self._unpack(best_result.x)
        self.log_likelihood_=-float(best_result.fun)

        all_z=np.concatenate(z_realizations) if sum(len(z) for z in z_realizations)>0 else np.array([])
        self.mean_mark_weight_=float(np.mean(np.exp(self.eta_*all_z))) if len(all_z)>0 else 1.0
        self.branching_ratio_=float(self.alpha_*self.mean_mark_weight_)
        self.is_stable_=bool(self.branching_ratio_<1.0)
        return self

    def score(self,events=None,marks=None,end_times=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations=self.events_
            z_realizations=self.z_marks_
            end_times=self.end_times_
        else:
            realizations,marks_realizations,end_times=self._prepare_realizations(events,marks,end_times)
            z_realizations=self._transform_marks_with_stats(marks_realizations,self.mark_stats_)

        theta=np.array([self.baseline_,self.alpha_,self.beta_,self.eta_],dtype=float)
        nll,_=self._nll_grad_all(theta,realizations,z_realizations,end_times)
        return -float(nll)

    def intensity_at_events(self,events=None,marks=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant intensity_at_events().")

        if events is None:
            times=self.events_[0]
            z=self.z_marks_[0]
        else:
            end_time=float(np.max(events)) if len(events) else 1.0
            realizations,marks_realizations,_=self._prepare_realizations(events,marks,end_time)
            if len(realizations)!=1:
                raise ValueError("intensity_at_events attend une seule réalisation.")
            times=realizations[0]
            z=self._transform_marks_with_stats(marks_realizations,self.mark_stats_)[0]

        mu,alpha,beta,eta=self.baseline_,self.alpha_,self.beta_,self.eta_
        intensities=np.zeros(len(times),dtype=float)
        r=0.0
        last_t=0.0
        n,k=len(times),0

        while k<n:
            t=times[k]
            dt=t-last_t
            if dt>0:
                r*=np.exp(-beta*dt)

            k2=k+1
            while k2<n and times[k2]==t:
                k2+=1

            lam=mu+alpha*r
            intensities[k:k2]=lam

            z_group=z[k:k2]
            w_group=np.exp(eta*z_group)
            r+=beta*np.sum(w_group)

            last_t=t
            k=k2

        return intensities

    def get_params(self):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline":self.baseline_,
            "alpha":self.alpha_,
            "beta":self.beta_,
            "eta":self.eta_,
            "mark_stats":dict(self.mark_stats_),
            "mean_mark_weight":self.mean_mark_weight_,
            "branching_ratio":self.branching_ratio_,
            "is_stable":self.is_stable_,
            "log_likelihood":self.log_likelihood_,
            "success":self.success_,
            "message":self.message_,
            "n_iter":self.n_iter_
        }
    
