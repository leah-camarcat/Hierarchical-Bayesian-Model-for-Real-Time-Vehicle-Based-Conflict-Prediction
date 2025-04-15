from pymc_experimental.model_builder import ModelBuilder
import pytensor.tensor as pt
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Union
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pytictoc import TicToc
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
from pyextremes import plot_mean_residual_life
from pyextremes import plot_parameter_stability
from scipy.stats import genpareto
import seaborn as sns
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

###################################### data pre-processing ######################################
data_test = pd.read_csv('total_data.csv')
data_test = data_test.iloc[:, 1:]
data_test.dropna(inplace=True)


y = data_test.MTTC
y = y[y < 5]

x_raw = data_test.drop(columns=['track_id', 'Frame', 'MTTC', 'egolat', 'egolon', 'abstime'])
roadsegs = x_raw['ild'].unique()
x_raw['ild'] = x_raw['ild'].map(str.strip)
nb_rs = len(roadsegs)
roadseg_lookup = dict(zip(sorted(roadsegs), range(nb_rs)))
x_raw["ild"] = x_raw["ild"].replace(roadseg_lookup).values

def dic(idata):
    'calculate deviance information criterion '
    llh_df = idata.log_likelihood

    d_theta = 0
    d_theta_bar = 0

    for i in llh_df.data_vars:
        llh = llh_df[i].values
        d_theta += -2 * np.mean(llh, axis=(0, 1)).sum()
        d_theta_bar += -2 * np.log(np.mean(np.exp(llh), axis=(0, 1))).sum()

    pd = d_theta - d_theta_bar
    dic = d_theta_bar + 2 * pd  # d_theta_bar + 2(d_theta - d_thetabar) = 2dtheta - dtheta_bar if negative dtheta_bar > d_theta
    return dic, pd


def estimate_gpd_params(data, threshold):
    """
    Estimate GPD parameters (shape, scale) for data exceeding a given threshold.
    """
    exceedances = data[data > threshold]
    if len(exceedances) < 10:
        return None, None  # Return None if insufficient data to fit reliably
    shape, loc, scale = genpareto.fit(exceedances, floc=0)  # Fit GPD
    return shape, scale, exceedances


def simulate_gpd_quantiles(shape, scale, sample_size, probabilities, m=100):
    """
    Simulate m independent samples from a GPD with given shape and scale,
    and compute the quantiles for specified probabilities.
    """
    simulated_quantiles = []
    for _ in range(m):
        sample = genpareto.rvs(shape, loc=0, scale=scale, size=sample_size)
        quantiles = np.quantile(sample, probabilities)
        simulated_quantiles.append(quantiles)
    # Average quantiles over m simulations
    return np.mean(simulated_quantiles, axis=0)


def calculate_square_error(observed_quantiles, simulated_quantiles):
    """
    Calculate the square error between observed and simulated quantiles.
    """
    return np.sum((simulated_quantiles / observed_quantiles) ** 2)


def find_optimal_threshold(data, thresholds, probabilities=np.arange(0.05, 1.0, 0.05), m=1000):
    """
    Find the optimal threshold for GPD tail modeling using the Square Error method.
    """
    square_errors = []
    
    for threshold in thresholds:
        # Step 1: Estimate GPD parameters for exceedances over the threshold
        shape, scale, exceedances = estimate_gpd_params(data, threshold)
        if shape is None or scale is None:
            square_errors.append(np.inf)
            continue
        
        # Step 2: Get observed quantiles for exceedances
        observed_quantiles = np.quantile(exceedances, probabilities)
        
        # Step 3: Simulate m samples from the GPD and calculate simulated quantiles
        simulated_quantiles = simulate_gpd_quantiles(shape, scale, len(exceedances), probabilities, m=m)
        
        # Step 4: Calculate square error between observed and simulated quantiles
        se = calculate_square_error(observed_quantiles, simulated_quantiles)
        square_errors.append(se)

    # Find the threshold with the minimum square error
    optimal_threshold = thresholds[np.argmin(square_errors)]
    return optimal_threshold, square_errors


def threshold_selection(y, thresholds):
     # MRL PLOT
    ym = np.array(-y[y < 4])
    date_index = pd.date_range(start='00:00', periods=len(ym), freq='min')
    dt = pd.Series(ym, index = date_index)
    plot_mean_residual_life(dt, extremes_type = 'low', thresholds=thresholds)
    plt.xlabel('Threshold (s)', fontsize=14)
    plt.ylabel('Mean excess', fontsize=14)
    plt.show()

    # # stability plot
    # plot_parameter_stability(dt, alpha=0.99, thresholds=np.linspace(-3,-1, 100))
    # plt.xlabel('Threshold (s)', fontsize=14)
    # plt.ylabel('Modified scale, $\sigma^*$', fontsize=14)
    # plt.show()

    # Optimisation method
    optimal_threshold, square_errors = find_optimal_threshold(-y, thresholds)

    # Plot the square errors across thresholds
    plt.plot(thresholds, square_errors, label="Square Error")
    plt.axvline(optimal_threshold, color="red", linestyle="--", label="Optimal Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Square Error")
    plt.title("Square Error for Threshold Selection")
    plt.legend()
    plt.show()

    print(f"Optimal Threshold: {optimal_threshold}")
    return optimal_threshold


def process_data(x_raw, y, threshold):
    negy = -y[y < -threshold].dropna()
    x = x_raw[x_raw.index.isin(negy.index)]
    negy = pd.Series(np.array(negy).reshape(len(negy),))

    idx = []
    scene_indices = x['Scene'].unique()
    for i in scene_indices:
        df = x[x.Scene == i]
        id = np.arange(len(df))
        idx.extend(id)
    print(len(idx))
    x['time_idx'] = idx

    print('len y', negy)
    print('len x', x)

    time = x.time_idx.values
    scaler = MinMaxScaler()
    S = np.array(x.Speed)
    F = np.array(x.Flow)
    K = F*12/S
    fig, ax = plt.subplots()
    plt.scatter(K, S)
    #plt.xlabel('Traffic flow (veh/5min)')
    plt.xlabel('Traffic density (veh/km)')
    plt.ylabel('Speed (km/h)')
    # sns.kdeplot(S)
    # plt.xlabel('Speed (km/h)')
    # plt.show()
    # fig, ax = plt.subplots()
    # sns.kdeplot(F)
    # plt.xlabel('Traffic flow (veh/5min)')
    plt.show()
    S = scaler.fit_transform(S.reshape(-1, 1)).flatten()
    F = scaler.fit_transform(F.reshape(-1, 1)).flatten()

    Sstd = np.array(x['Speed std'])
    Sstd = scaler.fit_transform(Sstd.reshape(-1, 1)).flatten()

    pos = np.array(x["distance"])
    pos = scaler.fit_transform(pos.reshape(-1, 1)).flatten()

    dV = np.array(x.speed)
    dV = scaler.fit_transform(dV.reshape(-1, 1)).flatten()

    dA = np.array(x.acceleration)
    dA = scaler.fit_transform(dA.reshape(-1, 1)).flatten()

    roadseg = x.ild
    roadsegs = x['ild'].unique()

    x_final = pd.concat((pd.Series(time), pd.Series(pos), pd.Series(dV), pd.Series(dA), pd.Series(S), pd.Series(F), pd.Series(Sstd), pd.Series(roadseg.values)), axis=1)
    x_final.columns = ['time', 'pos', 'dV', 'dA', 'S', 'F', 'Sstd', 'roadseg']

    print('describing x: ', x.describe)

    xtrain_pp = pd.DataFrame()
    ytrain_pp = pd.DataFrame()
    xtest_pp =  pd.DataFrame()
    ytest_pp =  pd.DataFrame()

    for i in np.unique(x_final.roadseg):
        df = x_final[x_final.iloc[:, -1] == i]
        if len(df) > 4:
            lentrain = 4*int(len(df)/5)
            print(i, lentrain)
        else:
            lentrain = 2*int(len(df)/3)
            print(i, lentrain)
        xtrain_pp = pd.concat([xtrain_pp, df.iloc[:lentrain, :]])
        ytrain_pp = pd.concat([ytrain_pp, negy[df.index[:lentrain]]])
        xtest_pp = pd.concat([xtest_pp, df.iloc[lentrain:, :]])
        ytest_pp = pd.concat([ytest_pp, negy[df.index[lentrain:]]])
    print(len(xtrain_pp), len(xtest_pp))
    return xtrain_pp, ytrain_pp, xtest_pp, ytest_pp


class GPDModel_veh(ModelBuilder):
    # Give the model a name
    _model_type = "GPDModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model
        """
        # Check the type of X and y and adjust access accordingly
        # X_values = X["input"].values
        X_values = X.values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.Data("x_data", X_values)
            y_data = pm.Data("y_data", y_values)
            
            # hyperpriors
            xi = pm.TruncatedNormal("xi", mu = 0, sigma = 0.5, lower=-1, upper=1)
            beta_phi = pm.Normal("beta_phi", mu = 0, sigma = 10, shape = 3)
            eps_phi_j = pm.Normal("eps_phi_j", mu = 0, sigma = 1)
            ti = pm.Normal("ti", mu = 0, sigma = 1)
            
            # Priors
            phi = eps_phi_j + ti*x_data[:, 0] \
                + beta_phi[0]*x_data[:, 1] + beta_phi[1]*x_data[:, 2]  + beta_phi[2]*x_data[:, 3]
                
            sigma = pm.math.exp(phi)

            # GPD log-likelihood
            def gpd_logp(value, mu, sigma, xi):
                scaled = (value - mu) / sigma
                logp_xi_nonzero = -pt.log(sigma) - (1 + 1/xi) * pt.log1p(xi * scaled) # from literatures and pymc github pull-request
                # logp_xi_nonzero = - n * pt.log(sigma) + (1/xi - 1) * pt.log(1 - xi * scaled)
                logp_xi_zero = -pt.log(sigma) - scaled
                return pt.switch(
                    pt.isclose(xi, 0), logp_xi_zero,
                    pt.switch((1 + xi * scaled) > 0, logp_xi_nonzero, -np.inf)
                )
            def gpd_random(mu, sigma, xi, rng=None, size=None):
                # sigma, xi = pm.distributions.draw_values([sigma, xi], point=point, size=size)
                return stats.genpareto.rvs(c=xi, loc=mu, scale=sigma, size=size, random_state=rng)
            
            obs = pm.CustomDist('y', mu, sigma, xi, logp = gpd_logp, random = gpd_random, shape=x_data.shape[0], observed = y_values)
            # obs = GeneralizedPareto('obs', mu, sigma, xi, observed = y_values)

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):
            #x_values = X["input"].values
            x_values = X.values
            print('len x values in data setter: ', len(x_values))
        else:
            # Assuming "input" is the first column
            x_values = X
        #x_values = X.values
        with self.model:
            pm.set_data({"x_data": x_values})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})
                print('x and y values', len(x_values), len(y.values) if isinstance(y, pd.Series) else len(y))
    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder
        """
        model_config: Dict = {
            "xi": 1,
            "beta_phi": 1,
            "eps_phi_j": 1,
            "ti": 1
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        """
        sampler_config: Dict = {
            "draws": 50_000,
            "tune": 10_000,
            "chains": 4,
            "target_accept": 0.9,
            "nuts_sampler": "blackjax",
            # "nuts": {"chain_method": "vectorized", "progressbar": False},
            "idata_kwargs":{"log_likelihood": True}
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        if isinstance(X, pd.DataFrame):
            timeindices = np.unique(X['time'])
            coords = {}
            coords['RoadSeg'] = np.unique(X.iloc[:, -1])
            coords['Time'] = np.unique(X.iloc[:, 0])
            coords['obs_id'] = np.arange(len(X))
        else:
            timeindices = np.unique(X[:,0])
            coords = {}
            coords['RoadSeg'] = np.unique(X[:, -1])
            coords['Time'] = np.unique(X[:, 0])
            coords['obs_id'] = np.arange(len(X))
        self.model_coords = coords
        print('len coords', len(coords['obs_id']))
        # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


class GPDModel_traff(ModelBuilder):
    # Give the model a name
    _model_type = "GPDModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model
        """
        # Check the type of X and y and adjust access accordingly
        # X_values = X["input"].values
        X_values = X.values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.Data("x_data", X_values)
            y_data = pm.Data("y_data", y_values)
            
            # hyperpriors
            xi = pm.TruncatedNormal("xi", mu = 0, sigma = 1, lower = -0.62, upper = 0.62)
            beta_phi = pm.Normal("beta_phi", mu = 0, sigma = 10, shape = 3)
            gamma_phi = pm.Normal("gamma_phi", mu = 0, sigma = 10, shape = 3)
            eps_phi_j = pm.Normal("eps_phi_j", mu = 0, sigma = 1)
            ti = pm.Normal("ti", mu = 0, sigma = 1)
            
            # Priors
            # scenario 1:
            phi = eps_phi_j + ti*x_data[:,0] \
                + beta_phi[0]*x_data[:,1] + beta_phi[1]*x_data[:,2]  + beta_phi[2]*x_data[:,3] \
                + gamma_phi[0]*x_data[:,4] + gamma_phi[1]*x_data[:,5] + gamma_phi[2]*x_data[:,6]
                
            sigma = pm.math.exp(phi)
            # mu = threshold

            # GPD log-likelihood
            def gpd_logp(value, mu, sigma, xi):
                scaled = (value - mu) / sigma
                logp_xi_nonzero = -pt.log(sigma) - (1 + 1/xi) * pt.log1p(xi * scaled) # from literatures and pymc github pull-request
                # logp_xi_nonzero = - n * pt.log(sigma) + (1/xi - 1) * pt.log(1 - xi * scaled)
                logp_xi_zero = -pt.log(sigma) - scaled
                return pt.switch(
                    pt.isclose(xi, 0), logp_xi_zero,
                    pt.switch((1 + xi * scaled) > 0, logp_xi_nonzero, -np.inf)
                )
            def gpd_random(mu, sigma, xi, rng=None, size=None):
                # sigma, xi = pm.distributions.draw_values([sigma, xi], point=point, size=size)
                return stats.genpareto.rvs(c=xi, loc=mu, scale=sigma, size=size, random_state=rng)
            
            obs = pm.CustomDist('y', mu, sigma, xi, logp = gpd_logp, random = gpd_random, shape=x_data.shape[0], observed = y_values)
            # obs = GeneralizedPareto('obs', mu, sigma, xi, observed = y_values)

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):
            #x_values = X["input"].values
            x_values = X.values
            print('len x values in data setter: ', len(x_values))
        else:
            # Assuming "input" is the first column
            x_values = X
        #x_values = X.values
        with self.model:
            pm.set_data({"x_data": x_values})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})
                print('x and y values', len(x_values), len(y.values) if isinstance(y, pd.Series) else len(y))
    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder
        """
        model_config: Dict = {
            "alpha_phi": 1,
            "xi": 1,
            "beta_phi": 1,
            "eps_phi_j": 1,
            "gamma_phi": 1
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        """
        sampler_config: Dict = {
            "draws": 50_000,
            "tune": 20_000,
            "chains": 4,
            "target_accept": 0.95,
            "nuts_sampler": "blackjax", 
            # "nuts": {"chain_method": "vectorized", "progressbar": False},
            "idata_kwargs":{"log_likelihood": True}
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        if isinstance(X, pd.DataFrame):
            timeindices = np.unique(X['time'])
            coords = {}
            coords['RoadSeg'] = np.unique(X.iloc[:, -1])
            coords['Time'] = np.unique(X.iloc[:, 0])
            coords['obs_id'] = np.arange(len(X))
        else:
            timeindices = np.unique(X[:,0])
            coords = {}
            coords['RoadSeg'] = np.unique(X[:, -1])
            coords['Time'] = np.unique(X[:, 0])
            coords['obs_id'] = np.arange(len(X))
        self.model_coords = coords
        print('len coords', len(coords['obs_id']))
        # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


class GPDModel_pp(ModelBuilder):
    # Give the model a name
    _model_type = "GPDModel_pp"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model
        """
        # Check the type of X and y and adjust access accordingly
        #X_values = X["input"].values
        X_values = X.values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)
        #RS = X_values[:,-1].astype(int)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.Data("x_data", X_values)
            y_data = pm.Data("y_data", y_values)
            #obs_id = pm.Data("obs_id", np.arange(len(X_values)))
            roadseg_idx = pm.Data("roadseg_idx", X_values[:,-1].astype(int), dims="obs_id")

            # hyperpriors
            xi = pm.TruncatedNormal("xi", mu = 0, sigma = 1, lower=-0.6, upper=0.6)
            beta_phi = pm.Normal("beta_phi", mu = 0, sigma = 10, shape = 4)
            alpha_phi = pm.Normal("alpha_phi", mu = 0, sigma = 10, shape = 2)
            gamma_phi = pm.Normal("gamma_phi", mu = 0, sigma = 10, shape=len(np.unique(X_values[:,-1].astype(int))))
            gamma_phi2 = pm.Normal("gamma_phi2", mu = 0, sigma = 10, shape=len(np.unique(X_values[:,-1].astype(int))))
            gamma_phi3 = pm.Normal("gamma_phi3", mu = 0, sigma = 10)
            eps_phi_j = pm.Normal("eps_phi_j", mu = 0, sigma = 1)
            ti = pm.Normal("ti", mu = 0, sigma = 1)
            
            # Priors
            phi = eps_phi_j + ti*x_data[:,0] \
            + beta_phi[0]*x_data[:,1] + beta_phi[1]*x_data[:,2]  + beta_phi[2]*x_data[:,3] \
            + gamma_phi[roadseg_idx]*x_data[:,4] + gamma_phi2[roadseg_idx]*x_data[:,5] + gamma_phi3*x_data[:,6] 
            sigma = pm.math.exp(phi)
            mu = threshold

            # GPD log-likelihood
            def gpd_logp(value, mu, sigma, xi):
                scaled = (value - mu) / sigma
                logp_xi_nonzero = -pt.log(sigma) - (1 + 1/xi) * pt.log1p(xi * scaled) # from literatures and pymc github pull-request
                logp_xi_zero = -pt.log(sigma) - scaled
                return pt.switch(
                    pt.isclose(xi, 0), logp_xi_zero,
                    pt.switch((1 + xi * scaled) > 0, logp_xi_nonzero, -np.inf)
                )
            def gpd_random(mu, sigma, xi, rng=None, size=None):
                # sigma, xi = pm.distributions.draw_values([sigma, xi], point=point, size=size)
                return stats.genpareto.rvs(c=xi, loc=mu, scale=sigma, size=size, random_state=rng)
            obs = pm.CustomDist('y', mu, sigma, xi, logp = gpd_logp, random = gpd_random, shape=x_data.shape[0], observed = y_values)
            # obs = GeneralizedPareto('obs', mu, sigma, xi, observed = y_values)

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):
            #x_values = X["input"].values
            x_values = X.values
            print('len x values in data setter: ', len(x_values))
        else:
            # Assuming "input" is the first column
            x_values = X
        #x_values = X.values
        self._generate_and_preprocess_model_data(x_values, y)
        with self.model:
            pm.set_data(
                {
                "x_data": x_values,
                "roadseg_idx": x_values[:, -1].astype(np.int32),
                },
                coords={
                     'RoadSeg': np.unique(x_values[:, -1]), 
                     'Time': np.unique(x_values[:, 0]), 
                     'obs_id': np.arange(len(x_values))
                     }
                )
                        #  "roadseg_idx": x_values[:, -1].astype(int),
                        #  "obs_id": np.arange(len(x_values))
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})
                print('x and y values', len(x_values), len(y.values) if isinstance(y, pd.Series) else len(y))

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder
        """
        model_config: Dict = {
            "xi": 1,
            "beta_phi": 1, 
            "gamma_phi": 1,
            "eps_phi_j": 1,
            "gamma_phi2": 1, 
            "gamma_phi3": 1,
            "ti": 1
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        """
        sampler_config: Dict = {
            "draws": 50_000,
            "tune": 10_000,
            "chains": 4,
            "target_accept": 0.95,
            "nuts_sampler": "blackjax", 
            # "nuts": {"chain_method": "vectorized", "progressbar": False},
            "idata_kwargs":{"log_likelihood": True}
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        if isinstance(X, pd.DataFrame):
            coords = {}
            coords['RoadSeg'] = np.unique(X.iloc[:, -1])
            coords['Time'] = np.unique(X.iloc[:, 0])
            coords['obs_id'] = np.arange(len(X))
        else:
            coords = {}
            coords['RoadSeg'] = np.unique(X[:, -1])
            coords['Time'] = np.unique(X[:, 0])
            coords['obs_id'] = np.arange(len(X))
        self.model_coords = coords
        print('len coords', len(coords['obs_id']), len(coords['RoadSeg']))
        # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y

# optimal_threshold = threshold_selection(y, thresholds=np.linspace(-3.8, -2, 50))

mu = -3.68
xtrain_pp, ytrain_pp, xtest_pp, ytest_pp = process_data(x_raw, y, mu)

plt.hist(ytrain_pp, bins=50, label="train data")
plt.hist(ytest_pp, bins=50, label='test data')
plt.ylabel('Count')
plt.xlabel('MTTC values')
plt.legend()
plt.show()

###################################### Fit the model VEH ######################################
t = TicToc()
t.tic()
model = GPDModel_veh()
idata = model.fit(xtrain_pp, pd.Series(np.array(ytrain_pp).reshape(len(ytrain_pp),)))
t.toc()

####### Evaluate the model: DIC, loo, rhat ########
dic_val, pd_dic = dic(idata)
print(f"DIC vehicle only: {dic_val}")
hierarchical_loo = az.loo(idata)
print(hierarchical_loo)
rhat = pm.rhat(idata)
print('r hat: ', rhat)
#print(az.summary(idata, round_to=2))

# ######## inference: mae, mse, cp ########
t = TicToc()
t.tic()
pred_mean = model.predict(xtest_pp)
pred_samples = model.predict_posterior(xtest_pp)
t.toc()

mae = np.mean(np.abs(pred_mean - np.array(ytest_pp)))
mse = np.mean((pred_mean - np.array(ytest_pp))**2)
print('mae: ', mae, 'mse: ', mse)

posterior = az.extract(idata, num_samples=len(xtest_pp))
sigma = np.exp(posterior["eps_phi_j"] + posterior["ti"]*xtest_pp.time) + posterior["beta_phi"][0]*xtest_pp.pos \
    + posterior["beta_phi"][1] * xtest_pp.dV + posterior["beta_phi"][2]*xtest_pp.dA
xi = posterior["xi"]
sigma = np.mean(sigma.values)
xi = np.mean(xi.values)
print('sigma: ', sigma, 'xi: ', xi)
pow = float(-1/xi)
bottom = float(1-xi*mu/sigma)
CP = bottom**(pow)
print('CP: ', CP)


###################################### Fit the model VEH+TRAFF ######################################
t = TicToc()
t.tic()
modelT = GPDModel_traff()
idataT = modelT.fit(xtrain_pp, pd.Series(np.array(ytrain_pp).reshape(len(ytrain_pp),)))
t.toc()

######## Evaluate the model: DIC, loo, rhat ########
dic_valT, pd_dicT = dic(idataT)
print(f"DIC with traffic: {dic_valT}")
hierarchical_looT = az.loo(idataT)
print(hierarchical_looT)
#print(az.summary(idataT, round_to=2))

######### inference: mae, mse, cp ########
t = TicToc()
t.tic()
pred_meanT = modelT.predict(xtest_pp)
pred_samplesT = modelT.predict_posterior(xtest_pp)
t.toc()

maeT = np.mean(np.abs(pred_meanT - np.array(ytest_pp)))
mseT = np.mean((pred_meanT - np.array(ytest_pp))**2)
print('mae: ', maeT, 'mse: ', mseT)

posterior = az.extract(idataT, num_samples=len(xtest_pp))
sigma = np.exp(posterior["eps_phi_j"] + posterior["ti"]*xtest_pp.time) \
               + posterior["beta_phi"][0]*xtest_pp.pos + posterior["beta_phi"][1]*xtest_pp.dV + posterior["beta_phi"][2]*xtest_pp.dA \
                + posterior["gamma_phi"][0]*xtest_pp.S + posterior["gamma_phi"][1]*xtest_pp.F + posterior["gamma_phi"][2]*xtest_pp.Sstd
xi = posterior["xi"]
sigmaT = np.mean(sigma.values)
xiT = np.mean(xi.values)
print('sigma: ', sigmaT, 'xi: ', xiT)
pow = float(-1/xiT)
bottom = float(1-xiT*mu/sigmaT)
CPT = bottom**(pow)
print('CP: ', CPT)


###################################### PARTIAL POOLING - Fit the model VEH+TRAFF ######################################

t = TicToc()
t.tic()
model_pp = GPDModel_pp()

#idata = model.fit(x_train, pd.Series(y_train.values.reshape(len(x_train),)))
idata_pp = model_pp.fit(pd.DataFrame(xtrain_pp), pd.Series(np.array(ytrain_pp).reshape(len(ytrain_pp),)))
t.toc()

dic_pp, pd_dic = dic(idata_pp)
print(f"DIC: {dic_pp}")

hierarchical_loo = az.loo(idata_pp)
print(hierarchical_loo)

# calculate CP 
posterior = az.extract(idata_pp, num_samples=len(xtest_pp))
sigma = np.exp(posterior["eps_phi_j"] + posterior["beta_phi"][0]*xtest_pp.pos +  posterior["beta_phi"][1]*xtest_pp.dV +  posterior["beta_phi"][2]*xtest_pp.dA + posterior["ti"]*xtest_pp.time) \
                      + posterior["gamma_phi"][0]*xtest_pp.S + posterior["gamma_phi"][1]*xtest_pp.F + posterior["gamma_phi"][2]*xtest_pp.Sstd 
xi = posterior["xi"]
sigma = np.mean(sigma.values)
xi = np.mean(xi.values)
print('sigma:', sigma, 'xi: ', xi)
pow = float(-1/xi)
bottom = float(1-xi*mu/sigma)
CP = bottom**(pow)
print('CP:', CP)
