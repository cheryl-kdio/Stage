import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import kstest


class BaseUnivariateHawkesMLE:
    param_names=("mu","alpha","beta")

    extra_arg_name=None
    extra_name=None
    extra_default=1.0
    extra_positive=False
    extra_standardize=False

    def __init__(
        self,
        beta_init=1.0,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        min_decay=1e-8,
        alpha_upper=None,
        beta_upper=None,
        alpha_l2=0.0,
        beta_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        self.beta_init=float(beta_init)
        self.max_iter=int(max_iter)
        self.tol=float(tol)
        self.min_baseline=float(min_baseline)
        self.min_decay=float(min_decay)
        self.alpha_upper=alpha_upper
        self.beta_upper=beta_upper
        self.alpha_l2=float(alpha_l2)
        self.beta_l2=float(beta_l2)
        self.eta_l2=float(eta_l2)
        self.n_starts=int(n_starts)
        self.random_state=random_state
        if self.beta_init<=0:
            raise ValueError("beta_init doit être strictement positif.")

    @staticmethod
    def _as_1d_float(x,name):
        x=np.asarray(x,dtype=float).ravel()
        if np.any(~np.isfinite(x)):
            raise ValueError(f"{name} doit contenir des valeurs finies.")
        return x

    def _prepare_one_realization(self,times,extra=None):
        times=self._as_1d_float(times,"times")

        if self.extra_name is None:
            order=np.argsort(times)
            return times[order]

        if extra is None:
            extra_arr=np.full_like(times,float(self.extra_default),dtype=float)
        else:
            extra_arr=self._as_1d_float(extra,self.extra_name)

        if times.shape!=extra_arr.shape:
            raise ValueError(f"times et {self.extra_name} doivent avoir la même longueur.")

        if self.extra_positive and np.any(extra_arr<=0):
            raise ValueError(f"Tous les {self.extra_name} doivent être strictement positifs.")

        order=np.argsort(times)
        return times[order],extra_arr[order]

    def _prepare_realizations(self,events,end_times=None,extra=None):
        realizations=[]
        extra_realizations=[] if self.extra_name is not None else None

        if isinstance(events,np.ndarray):
            if self.extra_name is None:
                realizations=[self._prepare_one_realization(events)]
            else:
                t,x=self._prepare_one_realization(events,extra)
                realizations,extra_realizations=[t],[x]

        elif isinstance(events,(list,tuple)):
            if len(events)==0:
                raise ValueError("events ne peut pas être vide.")

            if all(np.ndim(x)==0 for x in events):
                if self.extra_name is None:
                    realizations=[self._prepare_one_realization(events)]
                else:
                    t,x=self._prepare_one_realization(events,extra)
                    realizations,extra_realizations=[t],[x]

            else:
                if self.extra_name is None:
                    realizations=[self._prepare_one_realization(ev) for ev in events]
                else:
                    if extra is None:
                        extra_iter=[None]*len(events)
                    else:
                        if len(extra)!=len(events):
                            raise ValueError(
                                f"Pour plusieurs réalisations, {self.extra_name} doit avoir "
                                "la même longueur que events."
                            )
                        extra_iter=extra

                    for ev,ex in zip(events,extra_iter):
                        t,x=self._prepare_one_realization(ev,ex)
                        realizations.append(t)
                        extra_realizations.append(x)

        else:
            if self.extra_name is None:
                realizations=[self._prepare_one_realization(events)]
            else:
                t,x=self._prepare_one_realization(events,extra)
                realizations,extra_realizations=[t],[x]

        end_times=self._prepare_end_times(realizations,end_times)
        return realizations,extra_realizations,end_times

    @staticmethod
    def _prepare_end_times(realizations,end_times):
        if end_times is None:
            Ts=[]
            for t in realizations:
                if len(t)==0:
                    raise ValueError("end_times est requis si une réalisation est vide.")
                Ts.append(float(t[-1]))
            warnings.warn(
                "end_times non fourni : utilisation du dernier timestamp. "
                "Pour une MLE correcte, fournissez l'horizon réel d'observation.",
                RuntimeWarning,
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

        return np.asarray(Ts,dtype=float)

    def _prepare_fit_data(self,events,end_times=None,**kwargs):
        if self.extra_arg_name is None:
            realizations,_,end_times=self._prepare_realizations(events,end_times=end_times,extra=None)
            return realizations,None,end_times

        extra=kwargs.get(self.extra_arg_name,None)
        realizations,extra_realizations,end_times=self._prepare_realizations(
            events,
            end_times=end_times,
            extra=extra,
        )

        self._extra_raw_tmp_=extra_realizations
        payload=self._fit_extra_transform(extra_realizations)
        return realizations,payload,end_times

    def _prepare_score_data(self,events,end_times=None,**kwargs):
        if self.extra_arg_name is None:
            return self._prepare_fit_data(events,end_times=end_times)

        extra=kwargs.get(self.extra_arg_name,None)
        realizations,extra_realizations,end_times=self._prepare_realizations(
            events,
            end_times=end_times,
            extra=extra,
        )

        payload=self._score_extra_transform(extra_realizations)
        return realizations,payload,end_times

    def _fit_extra_transform(self,extra_realizations):
        if not self.extra_standardize:
            return extra_realizations
        payload,stats=self._fit_standardization(extra_realizations)
        self._extra_stats_tmp_=stats
        return payload

    def _score_extra_transform(self,extra_realizations):
        if not self.extra_standardize:
            return extra_realizations
        if not hasattr(self,"extra_stats_"):
            raise RuntimeError("Les statistiques de standardisation sont absentes.")
        return self._apply_standardization(extra_realizations,self.extra_stats_)

    @staticmethod
    def _fit_standardization(x_realizations):
        raw_all=[np.asarray(x,dtype=float).ravel() for x in x_realizations]
        concat=np.concatenate(raw_all) if sum(len(x) for x in raw_all)>0 else np.array([])

        if len(concat)==0:
            mean,std=0.0,1.0
        else:
            mean,std=float(np.mean(concat)),float(np.std(concat))
            if std<=1e-12:
                std=1.0

        z_realizations=[(x-mean)/std for x in raw_all]
        return z_realizations,{"mean":mean,"std":std}

    @staticmethod
    def _apply_standardization(x_realizations,stats):
        return [
            (np.asarray(x,dtype=float).ravel()-stats["mean"])/stats["std"]
            for x in x_realizations
        ]

    def _payload_at(self,payload,idx):
        return None if payload is None else payload[idx]

    def _unpack(self,theta):
        return {name:float(value) for name,value in zip(self.param_names,theta)}

    def _bounds(self):
        return [
            (self.min_baseline,None),
            (0.0,self.alpha_upper),
            (self.min_decay,self.beta_upper),
        ]

    def _initial_theta(self,realizations,end_times,payload=None):
        total_events=sum(len(x) for x in realizations)
        total_T=float(np.sum(end_times))

        mu0=max(
            0.5*total_events/max(total_T,1e-12),
            self.min_baseline*10,
        )
        alpha0=0.05
        beta0=max(self.beta_init,self.min_decay*10)

        if self.beta_upper is not None:
            beta0=min(beta0,self.beta_upper*0.9)

        return np.array([mu0,alpha0,beta0],dtype=float)

    def _random_start(self,theta0,rng):
        theta=theta0.copy()
        for i,name in enumerate(self.param_names):
            if name=="mu":
                theta[i]*=rng.lognormal(0.0,0.4)
            elif name in ("alpha","beta"):
                theta[i]*=rng.lognormal(0.0,0.7)
            elif name=="eta":
                theta[i]+=rng.normal(0.0,0.4)
        return theta

    @staticmethod
    def _project_into_bounds(theta,bounds):
        theta=theta.copy()
        for idx,(lo,hi) in enumerate(bounds):
            if lo is not None and theta[idx]<lo:
                theta[idx]=lo*10.0 if lo>0 else lo
            if hi is not None and theta[idx]>hi:
                theta[idx]=hi*0.9 if hi>0 else hi
        return theta

    def _nll_grad_one(self,theta,times,payload,T):
        raise NotImplementedError

    def _nll_grad_all(self,theta,realizations,payload,end_times):
        nll_total=0.0
        grad_total=np.zeros_like(theta,dtype=float)

        for idx,(times,T) in enumerate(zip(realizations,end_times)):
            nll,grad=self._nll_grad_one(
                theta,
                times,
                self._payload_at(payload,idx),
                float(T),
            )

            if not np.isfinite(nll):
                return np.inf,np.zeros_like(theta)

            nll_total+=nll
            grad_total+=grad

        return float(nll_total),grad_total

    def _add_l2(self,nll,grad,params):
        if self.alpha_l2>0:
            nll+=0.5*self.alpha_l2*params["alpha"]**2
            grad[self.param_names.index("alpha")]+=self.alpha_l2*params["alpha"]

        if self.beta_l2>0:
            nll+=0.5*self.beta_l2*params["beta"]**2
            grad[self.param_names.index("beta")]+=self.beta_l2*params["beta"]

        if "eta" in params and self.eta_l2>0:
            nll+=0.5*self.eta_l2*params["eta"]**2
            grad[self.param_names.index("eta")]+=self.eta_l2*params["eta"]

        return float(nll),grad

    def fit(self,events,end_times=None,x0=None,**kwargs):
        realizations,payload,end_times=self._prepare_fit_data(
            events,
            end_times=end_times,
            **kwargs,
        )

        if x0 is None:
            theta0=self._initial_theta(realizations,end_times,payload)
        else:
            theta0=np.asarray(x0,dtype=float).ravel()
            if theta0.size!=len(self.param_names):
                raise ValueError(
                    f"x0 doit avoir une longueur {len(self.param_names)} : "
                    f"{list(self.param_names)}."
                )

        bounds=self._bounds()
        rng=np.random.default_rng(self.random_state)
        best_result,best_fun=None,np.inf

        for start in range(max(1,self.n_starts)):
            start_theta=theta0.copy() if start==0 else self._random_start(theta0,rng)
            start_theta=self._project_into_bounds(start_theta,bounds)

            result=minimize(
                fun=lambda th:self._nll_grad_all(th,realizations,payload,end_times),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter":self.max_iter,"ftol":self.tol},
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
        self._payload_=payload
        self.log_likelihood_=-float(best_result.fun)

        params=self._unpack(best_result.x)
        self.baseline_=params["mu"]

        for name,value in params.items():
            setattr(self,f"{name}_",value)

        self._post_fit(payload)
        return self

    def _post_fit(self,payload):
        self.branching_ratio_=float(self.alpha_)
        self.is_stable_=bool(self.branching_ratio_<1.0)

    def score(self,events=None,end_times=None,**kwargs):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations=self.events_
            payload=self._payload_
            end_times=self.end_times_
        else:
            realizations,payload,end_times=self._prepare_score_data(
                events,
                end_times=end_times,
                **kwargs,
            )

        theta=np.array(
            [getattr(self,f"{name}_") for name in self.param_names],
            dtype=float,
        )

        nll,_=self._nll_grad_all(theta,realizations,payload,end_times)
        return -float(nll)

    def get_params(self):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        out={"baseline":self.baseline_}

        for name in self.param_names:
            if name!="mu":
                out[name]=getattr(self,f"{name}_")

        out.update({
            "log_likelihood":self.log_likelihood_,
            "success":self.success_,
            "message":self.message_,
            "n_iter":self.n_iter_,
        })

        if hasattr(self,"branching_ratio_"):
            out["branching_ratio"]=self.branching_ratio_
            out["is_stable"]=self.is_stable_

        if hasattr(self,"branching_ratio_empirical_"):
            out["mean_mark"]=self.mean_mark_
            out["branching_ratio_empirical"]=self.branching_ratio_empirical_
            out["is_stable_empirical"]=self.is_stable_empirical_

        if hasattr(self,"mark_stats_"):
            out["mark_stats"]=dict(self.mark_stats_)
            out["mean_mark_weight"]=self.mean_mark_weight_

        return out

    def _check_fitted(self):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant validation.")

    def _kernel_cumulative_at(self,t,times,payload):
        raise NotImplementedError

    def _kernel_intensity_at(self,t,times,payload):
        raise NotImplementedError

    def cumulative_intensity_at(self,times_eval=None,realization_idx=0):
        self._check_fitted()

        times=self.events_[realization_idx]
        payload=self._payload_at(self._payload_,realization_idx)

        scalar_input=np.ndim(times_eval)==0 and times_eval is not None

        if times_eval is None:
            grid=times.copy()
        elif scalar_input:
            grid=np.array([float(times_eval)],dtype=float)
        else:
            grid=np.asarray(times_eval,dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("times_eval doit contenir des valeurs finies.")
        if np.any(grid<0):
            raise ValueError("times_eval doit être positif.")

        Lambda=np.empty(len(grid),dtype=float)

        for i,t in enumerate(grid):
            Lambda[i]=self.baseline_*t+self.alpha_*self._kernel_cumulative_at(t,times,payload)

        return float(Lambda[0]) if scalar_input else Lambda

    def intensity_at(self,times_eval=None,realization_idx=0):
        self._check_fitted()

        times=self.events_[realization_idx]
        payload=self._payload_at(self._payload_,realization_idx)

        scalar_input=np.ndim(times_eval)==0 and times_eval is not None

        if times_eval is None:
            grid=times.copy()
        elif scalar_input:
            grid=np.array([float(times_eval)],dtype=float)
        else:
            grid=np.asarray(times_eval,dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("times_eval doit contenir des valeurs finies.")
        if np.any(grid<0):
            raise ValueError("times_eval doit être positif.")

        lam=np.empty(len(grid),dtype=float)

        for i,t in enumerate(grid):
            lam[i]=self.baseline_+self.alpha_*self._kernel_intensity_at(t,times,payload)

        return float(lam[0]) if scalar_input else lam

    def counting_process_at(self,times_eval=None,realization_idx=0):
        self._check_fitted()

        times=self.events_[realization_idx]
        scalar_input=np.ndim(times_eval)==0 and times_eval is not None

        if times_eval is None:
            grid=times.copy()
        elif scalar_input:
            grid=np.array([float(times_eval)],dtype=float)
        else:
            grid=np.asarray(times_eval,dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("times_eval doit contenir des valeurs finies.")
        if np.any(grid<0):
            raise ValueError("times_eval doit être positif.")

        N=np.searchsorted(times,grid,side="right").astype(float)
        return float(N[0]) if scalar_input else N

    def validation_processes(self,realization_idx=0,grid=None,n_grid=1000):
        self._check_fitted()

        times=self.events_[realization_idx]
        T=self.end_times_[realization_idx]

        if grid is None:
            grid=np.unique(np.r_[np.linspace(0.0,T,int(n_grid)),times])
        else:
            grid=np.asarray(grid,dtype=float).ravel()

        N=self.counting_process_at(grid,realization_idx=realization_idx)
        Lambda=self.cumulative_intensity_at(grid,realization_idx=realization_idx)
        M=N-Lambda
        lam=self.intensity_at(grid,realization_idx=realization_idx)

        return {"time":grid,"N":N,"Lambda":Lambda,"M":M,"lambda":lam}

    def martingale_at_events(self,realization_idx=0):
        self._check_fitted()

        times=self.events_[realization_idx]

        if len(times)==0:
            return {
                "event_times":times,
                "Lambda_events":np.array([]),
                "M_events":np.array([]),
                "tau":np.array([]),
                "martingale_increments":np.array([]),
            }

        if np.any(np.diff(times)==0):
            warnings.warn(
                "Présence de timestamps égaux. Les résidus de time-rescaling "
                "sont théoriquement valides pour un processus ponctuel simple.",
                RuntimeWarning,
            )

        Lambda_events=self.cumulative_intensity_at(times,realization_idx=realization_idx)
        N_events=np.arange(1,len(times)+1,dtype=float)
        M_events=N_events-Lambda_events
        tau=np.diff(np.r_[0.0,Lambda_events])
        martingale_increments=np.ones_like(tau)-tau

        return {
            "event_times":times,
            "Lambda_events":Lambda_events,
            "M_events":M_events,
            "tau":tau,
            "martingale_increments":martingale_increments,
        }

    def time_rescaling_residuals(self,realization_idx=None):
        self._check_fitted()

        indices=range(len(self.events_)) if realization_idx is None else [int(realization_idx)]

        all_tau=[]
        all_uniform=[]
        all_realization=[]
        all_event_index=[]

        for idx in indices:
            out=self.martingale_at_events(realization_idx=idx)
            tau=out["tau"]

            if len(tau)==0:
                continue

            uniform=1.0-np.exp(-tau)

            all_tau.append(tau)
            all_uniform.append(uniform)
            all_realization.append(np.full(len(tau),idx,dtype=int))
            all_event_index.append(np.arange(1,len(tau)+1,dtype=int))

        if len(all_tau)==0:
            return {
                "tau":np.array([]),
                "uniform":np.array([]),
                "realization":np.array([]),
                "event_index":np.array([]),
            }

        return {
            "tau":np.concatenate(all_tau),
            "uniform":np.concatenate(all_uniform),
            "realization":np.concatenate(all_realization),
            "event_index":np.concatenate(all_event_index),
        }

    def residual_ks_tests(self,realization_idx=None):
        residuals=self.time_rescaling_residuals(realization_idx=realization_idx)

        tau=residuals["tau"]
        u=residuals["uniform"]

        mask=np.isfinite(tau)&np.isfinite(u)&(tau>=0.0)&(u>=0.0)&(u<=1.0)
        tau=tau[mask]
        u=u[mask]

        if len(tau)==0:
            raise ValueError("Aucun résidu valide pour effectuer les tests.")

        ks_exp=kstest(tau,"expon")
        ks_uniform=kstest(u,"uniform")

        return {
            "n_residuals":int(len(tau)),
            "exp_ks_stat":float(ks_exp.statistic),
            "exp_pvalue":float(ks_exp.pvalue),
            "uniform_ks_stat":float(ks_uniform.statistic),
            "uniform_pvalue":float(ks_uniform.pvalue),
        }

    def plot_N_vs_Lambda(self,realization_idx=0,grid=None,n_grid=1000):
        import matplotlib.pyplot as plt

        self._check_fitted()

        times=self.events_[realization_idx]
        T=self.end_times_[realization_idx]

        if grid is None:
            grid=np.unique(np.r_[np.linspace(0.0,T,int(n_grid)),times])
        else:
            grid=np.asarray(grid,dtype=float).ravel()

        d=self.validation_processes(
            realization_idx=realization_idx,
            grid=grid,
            n_grid=n_grid,
        )

        fig,axes=plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(9,6),
            gridspec_kw={"height_ratios":[2,1]},
        )

        ax0,ax1=axes

        ax0.step(d["time"],d["N"],where="post",label="N(t)")
        ax0.plot(d["time"],d["Lambda"],label=r"$\hat{\Lambda}(t)$")
        ax0.set_ylabel("Valeur cumulée")
        ax0.set_title(r"$N(t)$ vs compensateur $\hat{\Lambda}(t)$")
        ax0.legend()
        ax0.grid(True)

        ax1.plot(d["time"],d["lambda"],label=r"$\hat{\lambda}(t)$")
        ax1.set_xlabel("t")
        ax1.set_ylabel("Intensité")
        ax1.set_title(r"Intensité estimée $\hat{\lambda}(t)$")
        ax1.legend()
        ax1.grid(True)

        fig.tight_layout()
        return fig,axes

    def plot_martingale(self,realization_idx=0,grid=None,n_grid=1000):
        import matplotlib.pyplot as plt

        d=self.validation_processes(realization_idx=realization_idx,grid=grid,n_grid=n_grid)

        fig,ax=plt.subplots()
        ax.step(d["time"],d["M"],where="post",label=r"$M(t)=N(t)-\hat{\Lambda}(t)$")
        ax.axhline(0.0,linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("Martingale compensée")
        ax.set_title(r"Diagnostic martingale $M(t)$")
        ax.legend()
        ax.grid(True)

        return fig,ax

    def plot_residuals_qq(self,realization_idx=None):
        import matplotlib.pyplot as plt

        residuals=self.time_rescaling_residuals(realization_idx=realization_idx)
        tau=residuals["tau"]
        tau=tau[np.isfinite(tau)&(tau>=0.0)]

        if len(tau)==0:
            raise ValueError("Aucun résidu valide à tracer.")

        tau_sorted=np.sort(tau)
        n=len(tau_sorted)
        probs=(np.arange(1,n+1)-0.5)/n
        exp_quantiles=-np.log(1.0-probs)

        fig,ax=plt.subplots()
        ax.scatter(exp_quantiles,tau_sorted)

        lim_min=min(float(np.min(exp_quantiles)),float(np.min(tau_sorted)))
        lim_max=max(float(np.max(exp_quantiles)),float(np.max(tau_sorted)))

        ax.plot([lim_min,lim_max],[lim_min,lim_max],linestyle="--")
        ax.set_xlabel("Quantiles théoriques Exp(1)")
        ax.set_ylabel("Résidus observés")
        ax.set_title("QQ-plot des résidus de time-rescaling")
        ax.grid(True)

        return fig,ax

    def plot_uniform_residuals(self,realization_idx=None):
        import matplotlib.pyplot as plt

        residuals=self.time_rescaling_residuals(realization_idx=realization_idx)
        u=residuals["uniform"]
        u=u[np.isfinite(u)&(u>=0.0)&(u<=1.0)]

        if len(u)==0:
            raise ValueError("Aucun résidu uniformisé valide à tracer.")

        u_sorted=np.sort(u)
        n=len(u_sorted)
        empirical=np.arange(1,n+1)/n

        fig,ax=plt.subplots()
        ax.step(u_sorted,empirical,where="post",label="CDF empirique")
        ax.plot([0.0,1.0],[0.0,1.0],linestyle="--",label="CDF Uniform(0,1)")
        ax.set_xlabel(r"$U_i=1-\exp(-\tau_i)$")
        ax.set_ylabel("CDF")
        ax.set_title("Résidus uniformisés")
        ax.legend()
        ax.grid(True)

        return fig,ax
    
class UnivariateHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes exponentiel classique.

    lambda(t)=mu+sum_{t_k<t} alpha*beta*exp(-beta*(t-t_k))
    """

    param_names=("mu","alpha","beta")
    extra_arg_name=None
    extra_name=None

    def fit(self,events,end_times=None,x0=None):
        return super().fit(events,end_times=end_times,x0=x0)

    def score(self,events=None,end_times=None):
        return super().score(events,end_times=end_times)

    @staticmethod
    def _kernel_integral_and_grad(times,T,beta):
        if len(times)==0:
            return 0.0,0.0

        rem=T-times
        e=np.exp(-beta*rem)
        S=np.sum(1.0-e)
        S_beta=np.sum(rem*e)

        return float(S),float(S_beta)

    def _nll_grad_one(self,theta,times,payload,T):
        p=self._unpack(theta)
        mu,alpha,beta=p["mu"],p["alpha"],p["beta"]

        if mu<=0 or alpha<0 or beta<=0:
            return np.inf,np.zeros_like(theta)

        n=len(times)
        r=np.zeros(n,dtype=float)
        q=np.zeros(n,dtype=float)

        if n>1:
            dt=np.diff(times)

            if np.any(dt<=0):
                raise RuntimeError("Les timestamps doivent être strictement croissants.")

            e=np.exp(-beta*dt)

            for k in range(1,n):
                r[k]=e[k-1]*(r[k-1]+beta)
                q[k]=e[k-1]*(q[k-1]+1.0-dt[k-1]*(r[k-1]+beta))

        lam=mu+alpha*r

        if np.any(lam<=0) or np.any(~np.isfinite(lam)):
            return np.inf,np.zeros_like(theta)

        inv_lam=1.0/lam
        S,S_beta=self._kernel_integral_and_grad(times,T,beta)

        ll=np.sum(np.log(lam))-mu*T-alpha*S
        grad_mu=np.sum(inv_lam)-T
        grad_alpha=np.sum(r*inv_lam)-S
        grad_beta=np.sum(alpha*q*inv_lam)-alpha*S_beta

        nll=-ll
        grad=np.array([-grad_mu,-grad_alpha,-grad_beta],dtype=float)

        return self._add_l2(nll,grad,p)

    def intensity_at_events(self,events=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant intensity_at_events().")

        if events is None:
            times=self.events_[0]
        else:
            times=self._prepare_one_realization(events)

        n=len(times)
        r=np.zeros(n,dtype=float)
        if n>1:
            dt=np.diff(times)
            if np.any(dt<=0):
                raise RuntimeError("Les timestamps doivent être strictement croissants.")

            e=np.exp(-self.beta_*dt)
            for k in range(1,n):
                r[k]=e[k-1]*(r[k-1]+self.beta_)

        return self.baseline_+self.alpha_*r

    def _kernel_cumulative_at(self,t,times,payload):
        if len(times)==0 or t<=0:
            return 0.0

        past=times[times<t]

        if len(past)==0:
            return 0.0

        return float(np.sum(1.0-np.exp(-self.beta_*(t-past))))

    def _kernel_intensity_at(self,t,times,payload):
        if len(times)==0 or t<=0:
            return 0.0

        past=times[times<t]

        if len(past)==0:
            return 0.0

        return float(np.sum(self.beta_*np.exp(-self.beta_*(t-past))))

class UnivariateMarkedAmplitudeHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes avec mark sur l'amplitude.

    lambda(t)=mu+sum_{t_k<t} alpha*exp(eta*z_k)*beta*exp(-beta*(t-t_k))

    Les marks sont standardisés :
        z_k=(mark_k-mean)/std
    """

    param_names=("mu","alpha","beta","eta")

    extra_arg_name="marks"
    extra_name="marks"
    extra_default=1.0
    extra_positive=False
    extra_standardize=True

    def __init__(
        self,
        beta_init=1.0,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        min_decay=1e-8,
        alpha_upper=None,
        beta_upper=None,
        eta_bounds=(-5.0,5.0),
        alpha_l2=0.0,
        beta_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        super().__init__(
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            min_baseline=min_baseline,
            min_decay=min_decay,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
            alpha_l2=alpha_l2,
            beta_l2=beta_l2,
            eta_l2=eta_l2,
            n_starts=n_starts,
            random_state=random_state,
        )
        self.eta_bounds=eta_bounds

    def fit(self,events,marks=None,end_times=None,x0=None):
        return super().fit(events,end_times=end_times,x0=x0,marks=marks)

    def score(self,events=None,marks=None,end_times=None):
        return super().score(events,end_times=end_times,marks=marks)

    def _bounds(self):
        return super()._bounds()+[self.eta_bounds]

    def _initial_theta(self,realizations,end_times,payload=None):
        base=super()._initial_theta(realizations,end_times,payload)
        return np.r_[base,0.0]

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
        p=self._unpack(theta)
        mu,alpha,beta,eta=p["mu"],p["alpha"],p["beta"],p["eta"]

        if mu<=0 or alpha<0 or beta<=0:
            return np.inf,np.zeros_like(theta)

        n=len(times)
        r=np.zeros(n,dtype=float)
        q=np.zeros(n,dtype=float)
        u=np.zeros(n,dtype=float)

        if n>1:
            dt=np.diff(times)

            if np.any(dt<=0):
                raise RuntimeError("Les timestamps doivent être strictement croissants.")

            e=np.exp(-beta*dt)
            w=np.exp(eta*z)

            for k in range(1,n):
                r[k]=e[k-1]*(r[k-1]+beta*w[k-1])
                q[k]=e[k-1]*(q[k-1]+w[k-1]-dt[k-1]*(r[k-1]+beta*w[k-1]))
                u[k]=e[k-1]*(u[k-1]+beta*z[k-1]*w[k-1])

        lam=mu+alpha*r

        if np.any(lam<=0) or np.any(~np.isfinite(lam)):
            return np.inf,np.zeros_like(theta)

        inv_lam=1.0/lam
        S,S_eta,S_beta=self._kernel_integral_and_grads(times,z,T,beta,eta)

        ll=np.sum(np.log(lam))-mu*T-alpha*S
        grad_mu=np.sum(inv_lam)-T
        grad_alpha=np.sum(r*inv_lam)-S
        grad_beta=np.sum(alpha*q*inv_lam)-alpha*S_beta
        grad_eta=np.sum(alpha*u*inv_lam)-alpha*S_eta

        nll=-ll
        grad=np.array([-grad_mu,-grad_alpha,-grad_beta,-grad_eta],dtype=float)

        return self._add_l2(nll,grad,p)

    def _post_fit(self,payload):
        self.marks_=self._extra_raw_tmp_
        self.z_marks_=payload
        self.mark_stats_=self._extra_stats_tmp_
        self.extra_stats_=self._extra_stats_tmp_

        all_z=np.concatenate(payload) if sum(len(z) for z in payload)>0 else np.array([])

        self.mean_mark_weight_=float(np.mean(np.exp(self.eta_*all_z))) if len(all_z)>0 else 1.0
        self.branching_ratio_=float(self.alpha_*self.mean_mark_weight_)
        self.is_stable_=bool(self.branching_ratio_<1.0)

    def intensity_at_events(self,events=None,marks=None):
        if not hasattr(self,"baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant intensity_at_events().")

        if events is None:
            times=self.events_[0]
            return self.intensity_at(times_eval=times,realization_idx=0)

        end_time=float(np.max(events)) if len(events) else 1.0
        realizations,z_realizations,_=self._prepare_score_data(
            events,
            end_times=end_time,
            marks=marks,
        )

        if len(realizations)!=1:
            raise ValueError("intensity_at_events attend une seule réalisation.")

        times=realizations[0]
        z=z_realizations[0]

        n=len(times)
        r=np.zeros(n,dtype=float)

        if n>1:
            dt=np.diff(times)

            if np.any(dt<=0):
                raise RuntimeError("Les timestamps doivent être strictement croissants.")

            e=np.exp(-self.beta_*dt)
            w=np.exp(self.eta_*z)

            for k in range(1,n):
                r[k]=e[k-1]*(r[k-1]+self.beta_*w[k-1])

        return self.baseline_+self.alpha_*r

    def _kernel_cumulative_at(self,t,times,payload):
        if len(times)==0 or t<=0:
            return 0.0

        z=payload
        mask=times<t

        if not np.any(mask):
            return 0.0

        past=times[mask]
        past_z=z[mask]
        w=np.exp(self.eta_*past_z)

        return float(np.sum(w*(1.0-np.exp(-self.beta_*(t-past)))))

    def _kernel_intensity_at(self,t,times,payload):
        if len(times)==0 or t<=0:
            return 0.0

        z=payload
        mask=times<t

        if not np.any(mask):
            return 0.0

        past=times[mask]
        past_z=z[mask]
        w=np.exp(self.eta_*past_z)

        return float(np.sum(self.beta_*w*np.exp(-self.beta_*(t-past))))
    

class MultivariateHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes exponentiel multivarié.

    Pour i,j = 0,...,d-1 :
        lambda_i(t) = mu_i + sum_j sum_{t_k^j < t}
                      alpha_ij * beta_ij * exp(-beta_ij * (t - t_k^j))

    Convention : alpha_ij mesure l'effet d'un événement de type j
    sur l'intensité de type i.

    Les données sont données sous forme :
        events = times
        types  = types associés aux événements

    Pour plusieurs réalisations intraday :
        events = [times_day_1, ..., times_day_R]
        types  = [types_day_1, ..., types_day_R]
        end_times = [T_1, ..., T_R]
    """

    extra_arg_name = "types"
    extra_name = "types"
    extra_default = 0
    extra_positive = False
    extra_standardize = False

    def __init__(self, n_types, **kwargs):
        super().__init__(**kwargs)

        self.n_types = int(n_types)
        if self.n_types < 1:
            raise ValueError("n_types doit être supérieur ou égal à 1.")

        d = self.n_types

        # La base utilise len(self.param_names) pour vérifier la taille de x0
        # et _random_start() pour perturber les paramètres. On répète donc
        # les noms par bloc : d mu, d*d alpha, d*d beta.
        self.param_names = (
            ("mu",) * d
            + ("alpha",) * (d * d)
            + ("beta",) * (d * d)
        )

    def fit(self, events, types=None, end_times=None, x0=None):
        if types is None:
            raise ValueError("types est requis pour le Hawkes multivarié.")
        return super().fit(events, end_times=end_times, x0=x0, types=types)

    def score(self, events=None, types=None, end_times=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations = self.events_
            payload = self._payload_
            end_times = self.end_times_

        else:
            if types is None:
                raise ValueError("types est requis pour scorer de nouvelles données.")
            realizations, payload, end_times = self._prepare_score_data(
                events,
                end_times=end_times,
                types=types,
            )

        theta = np.r_[self.mu_, self.alpha_.ravel(), self.beta_.ravel()]
        nll, _ = self._nll_grad_all(theta, realizations, payload, end_times)
        return -float(nll)

    def _unpack(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()
        d = self.n_types
        expected = d + 2 * d * d

        if theta.size != expected:
            raise ValueError(f"theta doit avoir une longueur {expected}.")

        idx = 0
        mu = theta[idx:idx + d]
        idx += d

        alpha = theta[idx:idx + d * d].reshape(d, d)
        idx += d * d

        beta = theta[idx:idx + d * d].reshape(d, d)

        return {
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
        }

    def _bounds(self):
        d = self.n_types
        return (
            [(self.min_baseline, None)] * d
            + [(0.0, self.alpha_upper)] * (d * d)
            + [(self.min_decay, self.beta_upper)] * (d * d)
        )

    def _initial_theta(self, realizations, end_times, payload=None):
        d = self.n_types
        total_T = float(np.sum(end_times))

        counts = np.zeros(d, dtype=float)

        if payload is not None:
            for ty in payload:
                ty = np.asarray(ty, dtype=float).ravel()

                if len(ty) == 0:
                    continue

                if np.any(ty != np.floor(ty)):
                    raise ValueError("types doit contenir des entiers.")

                ty = ty.astype(int)

                if np.any((ty < 0) | (ty >= d)):
                    raise ValueError("types doit être compris entre 0 et n_types-1.")

                counts += np.bincount(ty, minlength=d)[:d]
        else:
            total_events = sum(len(x) for x in realizations)
            counts[:] = total_events / max(d, 1)

        mu0 = np.maximum(
            0.5 * counts / max(total_T, 1e-12),
            self.min_baseline * 10.0,
        )

        # Initialisation faible pour éviter de partir trop près d'un régime explosif.
        # Une matrice pleine constante c/d a un rayon spectral environ c.
        alpha0 = np.full((d, d), 0.05 / max(d, 1), dtype=float)

        if self.alpha_upper is not None:
            alpha0 = np.minimum(alpha0, 0.9 * self.alpha_upper)

        beta0_scalar = max(self.beta_init, self.min_decay * 10.0)

        if self.beta_upper is not None:
            beta0_scalar = min(beta0_scalar, 0.9 * self.beta_upper)

        beta0 = np.full((d, d), beta0_scalar, dtype=float)

        return np.r_[mu0, alpha0.ravel(), beta0.ravel()]
    
    def _nll_grad_one(self, theta, times, types, T):
        p = self._unpack(theta)
        mu, A, B = p["mu"], p["alpha"], p["beta"]
        d = self.n_types
        times = np.asarray(times, dtype=float).ravel()

        if types is None:
            raise ValueError("types est requis.")

        types = np.asarray(types, dtype=float).ravel()
        if len(types) != len(times):
            raise ValueError("times et types doivent avoir la même longueur.")
        if np.any(types != np.floor(types)):
            raise ValueError("types doit contenir des entiers.")

        types = types.astype(int)
        if np.any((types < 0) | (types >= d)):
            raise ValueError("types doit être compris entre 0 et n_types-1.")
        if np.any(times < 0):
            raise ValueError("Les timestamps doivent être positifs ou nuls.")
        if np.any(times > T):
            return np.inf, np.zeros_like(theta)
        if np.any(mu <= 0) or np.any(A < 0) or np.any(B <= 0):
            return np.inf, np.zeros_like(theta)

        R = np.zeros((d, d), dtype=float)
        Q = np.zeros((d, d), dtype=float)

        grad_mu_log = np.zeros(d, dtype=float)
        grad_A_log = np.zeros((d, d), dtype=float)
        grad_B_log = np.zeros((d, d), dtype=float)

        log_term = 0.0

        for k, (t, m) in enumerate(zip(times, types)):
            if k > 0:
                dt = float(times[k] - times[k - 1])

                if dt <= 0:
                    raise RuntimeError("Les timestamps doivent être strictement croissants.")

                m_prev = types[k - 1]

                E = np.exp(-B * dt)

                R_old = R.copy()
                Q_old = Q.copy()

                add_R = np.zeros((d, d), dtype=float)
                add_Q = np.zeros((d, d), dtype=float)

                # L'événement précédent était de type m_prev.
                # Il excite toutes les intensités i via la colonne m_prev.
                add_R[:, m_prev] = B[:, m_prev]
                add_Q[:, m_prev] = 1.0

                # Analogue multivarié de :
                # r[k] = e[k-1] * (r[k-1] + beta)
                R = E * (R_old + add_R)
                
                # Analogue multivarié de :
                # q[k] = e[k-1] * (
                #     q[k-1] + 1.0 - dt[k-1] * (r[k-1] + beta)
                # )
                Q = E * (Q_old + add_Q - dt * (R_old + add_R))

            # Intensité juste avant l'événement courant t_k.
            lam = mu + np.sum(A * R, axis=1)
            lm = lam[m]

            if lm <= 0 or not np.isfinite(lm):
                return np.inf, np.zeros_like(theta)

            inv_lm = 1.0 / lm
            log_term += np.log(lm)

            grad_mu_log[m] += inv_lm
            grad_A_log[m, :] += R[m, :] * inv_lm
            grad_B_log[m, :] += A[m, :] * Q[m, :] * inv_lm

        compensator = T * np.sum(mu)

        grad_mu_comp = np.full(d, T, dtype=float)
        grad_A_comp = np.zeros((d, d), dtype=float)
        grad_B_comp = np.zeros((d, d), dtype=float)

        for j in range(d):
            tj = times[types == j]

            if len(tj) == 0:
                continue

            rem = T - tj

            if np.any(rem < 0):
                return np.inf, np.zeros_like(theta)

            E = np.exp(-B[:, [j]] * rem[None, :])

            S_A = np.sum(1.0 - E, axis=1)
            S_B = np.sum(rem[None, :] * E, axis=1)

            compensator += np.sum(A[:, j] * S_A)

            grad_A_comp[:, j] += S_A
            grad_B_comp[:, j] += A[:, j] * S_B

        ll = log_term - compensator

        grad_mu = grad_mu_comp - grad_mu_log
        grad_A = grad_A_comp - grad_A_log
        grad_B = grad_B_comp - grad_B_log

        nll = -float(ll)

        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * float(np.sum(A * A))
            grad_A += self.alpha_l2 * A

        if self.beta_l2 > 0:
            nll += 0.5 * self.beta_l2 * float(np.sum(B * B))
            grad_B += self.beta_l2 * B

        grad = np.r_[grad_mu, grad_A.ravel(), grad_B.ravel()]

        return nll, grad

    def _post_fit(self, payload):
        self.types_ = payload
        self.branching_matrix_ = self.alpha_
        self.branching_ratio_ = float(np.max(np.abs(np.linalg.eigvals(self.alpha_))))
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

    def get_params(self):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline": self.baseline_,
            "mu": self.mu_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "branching_matrix": self.branching_matrix_,
            "branching_ratio": self.branching_ratio_,
            "is_stable": self.is_stable_,
            "log_likelihood": self.log_likelihood_,
            "success": self.success_,
            "message": self.message_,
            "n_iter": self.n_iter_,
        }
