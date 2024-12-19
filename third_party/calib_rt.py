

import numpy as np
import networkx as nx
import statsmodels.api as sm
from scipy.interpolate import interp1d

def screen_by_hist(x_data, y_data, bins):
    extenti = (x_data.min()-0.1, x_data.max()+0.1)
    extentj = (y_data.min()-0.1, y_data.max()+0.1)
    hist, edges_x, edges_y = np.histogram2d(
        x_data, y_data, bins=bins,range=(extenti, extentj)
    )
    edges_x = (edges_x[:-1] + edges_x[1:]) / 2
    edges_y = (edges_y[:-1] + edges_y[1:]) / 2
    
    cell_idXy = np.stack((np.arange(hist.shape[0]),hist.argmax(axis=1)),axis=-1)
    cell_idYx = np.stack((hist.argmax(axis=0),np.arange(hist.shape[1])),axis=-1)
    cell_idxy = np.vstack((cell_idXy,cell_idYx))
    cell_idxy = np.unique(cell_idxy,axis=0)
    cell_idxy = cell_idxy[~(cell_idxy == 0).any(axis=1)]
    
    x_hist = edges_x[cell_idxy[:,0]]
    y_hist = edges_y[cell_idxy[:,1]]
    cell_counts = hist[tuple(cell_idxy.T)]

    rho=cell_counts
    if np.max(rho)-np.min(rho)!=0:
        rho = (rho - np.min(rho))/(np.max(rho)-np.min(rho))+1
    return x_hist, y_hist, rho

def screen_by_graph(x_screen1, y_screen1, rho):
    G = nx.DiGraph()
    for i in range(len(x_screen1)):
        x_curr, y_curr = x_screen1[i], y_screen1[i]
        candidates_idx = (x_screen1 >= x_curr) & (y_screen1 >= y_curr)
        candidates_idx[i] = False
        
        x_candidates = x_screen1[candidates_idx]
        y_candidates = y_screen1[candidates_idx]
        rho_candidates = rho[candidates_idx]

        candidates = [((x_curr, y_curr), (x, y),{"edge":rho[i]*r/((x_curr-x)**2+(y_curr-y)**2)**0.5}) 
                      for x, y ,r in zip(x_candidates, y_candidates, rho_candidates)]
        G.add_edges_from(candidates)

    longest_path = nx.dag_longest_path(G,weight="edge")
    x_screen2, y_screen2 = zip(*longest_path)
    x_screen2, y_screen2 = np.array(x_screen2), np.array(y_screen2)

    return x_screen2, y_screen2

def polish_ends(x_screen2, y_screen2, tol_bins):
    center_idx = int(len(x_screen2) / 4)

    x, y = x_screen2[:center_idx], y_screen2[:center_idx]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    
    break_idx = 0
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx < len(x) * 0.25)[0]
        if len(idx) > 0:
            break_idx = breaks_idx[idx][-1] + 1
    x_left, y_left = x[break_idx:], y[break_idx:]

    x, y = x_screen2[center_idx:], y_screen2[center_idx:]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    
    break_idx = len(x)
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx > len(x) * 0.75)[0] 
        if len(idx) > 0:
            break_idx = breaks_idx[idx][0] + 1 
    x_right, y_right = x[:break_idx], y[:break_idx]
    
    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    return x, y

class InputDataError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def fit_by_lowess(x, y, manual_frac=-1):
    data=np.column_stack((x,y))
    data = data[data[:, 0] != 0]
    x,y=data[:,0],data[:,1]

    if manual_frac == -1:
        frac=choose_frac(x,y)
        print(f"Choose lowess frac: {frac:.2f}")
    else:
        frac=manual_frac

    # by lowess
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    return x_fit, y_fit

def choose_frac(x, y):
    frac_v = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    mape_v = []
    for frac in frac_v:
        y_pred = sm.nonparametric.lowess(y, x, frac, return_sorted=False)
        mape = np.nanmean(np.abs(y - y_pred) / y)
        mape_v.append(mape)
    return frac_v[np.argmin(mape_v)]


class Predictor:
    def __init__(self,x_fit,y_fit) -> None:
        
        data=self.__make_x_unique(x_fit,y_fit)
        x_fit,y_fit=data[:,0],data[:,1]
        self.f_in = interp1d(x_fit, y_fit, kind ='linear')

        # extrapolate for start
        idx_start = np.argsort(x_fit)[:5]
        coefficients = np.polyfit(x_fit[idx_start], y_fit[idx_start], 1)
        self.f_start = np.poly1d(coefficients)

        # extrapolate for end
        idx_end = np.argsort(x_fit)[::-1][:5]
        coefficients = np.polyfit(x_fit[idx_end], y_fit[idx_end], 1)
        self.f_end = np.poly1d(coefficients)

    def __make_x_unique(self,x_fit,y_fit):
        data=np.column_stack((x_fit,y_fit))
        unique_data=dict()
        for x,y in data:
            if not unique_data.get(x):
                unique_data[x]=y
        return np.array(list(unique_data.items()))

    def predict(self,x_interp):
        x_max, x_min = self.f_in.x.max(), self.f_in.x.min()
        is_inner = (x_interp <= x_max) & (x_interp >= x_min)
        is_start = x_interp < x_min
        is_end = x_interp > x_max
        x_in = x_interp[is_inner]
        x_start = x_interp[is_start]
        x_end = x_interp[is_end]
        x_in_pred = self.f_in(x_in)
        x_start_pred = self.f_start(x_start)
        x_end_pred = self.f_end(x_end)
        pred = np.empty_like(x_interp)
        pred[is_inner] = x_in_pred
        pred[is_start] = x_start_pred
        pred[is_end] = x_end_pred
        return pred

def cal_mrd(y, y_pred):
    return np.nanmean(np.abs(y - y_pred) / y)


def fit_by_raw_lowess(x, y,frac=0.1):
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    return x_fit, y_fit


class Normalization(object):
    def __init__(self,x,y) -> None:
        self.params = {"x":(np.min(x),np.max(x)-np.min(x)),
                       "y":(np.min(y),np.max(y)-np.min(y))}
        self.x_normal = self.normal("x",x)
        self.y_normal = self.normal("y",y)

    def normal(self,key,x):
        keymin,keylen = self.params[key]
        return (x-keymin)/keylen
    
    def denormal(self,key,x):
        keymin,keylen = self.params[key]
        return keylen*x+keymin
    
    def get_normalized_data(self):
        return self.x_normal,self.y_normal
    
    def denormalize_data(self,x_normal,y_normal):
        x = self.denormal(x_normal)
        y = self.denormal(y_normal)
        return x,y

class Calib_RT(object):
    """
    Calib_RT is used for RT calibration.

    Attributes:
        fit(): fit by Calib_RT model
        predict(): predict by Calib_RT model
        setdata(): Set data for Calib_RT model
    """
    
    def __init__(self,bins:int=100,tol_bins:float=10) -> None:
        """
        Set params for Calib_RT model

        Args:
            bins(int): 
            tol_bins(float):
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
        """
        self.bins = bins
        self.tol_bins = tol_bins

    def setdata(self,x,y) -> None:
        """
        Set data for Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT
            y(array_like): 1-D array_like for Measured RT
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
            >>> model.setdata(x,y)
        """
        self.x = x
        self.y = y
        self.nomral = Normalization(x,y)
        self.x_normal,self.y_normal = self.nomral.get_normalized_data()

    def fit(self,x,y,manual_frac:float=-1) -> None:
        """
        fit Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT
            y(array_like): 1-D array_like for Measured RT
            manual_frac(float): 
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=20)
            >>> model.fit(x,y)
        """
        self.setdata(x,y)
        self.__fit(manual_frac)

    def __fit(self,manual_frac):

        x_hist,y_hist,rho_hist = screen_by_hist(self.x_normal,
                                                self.y_normal,
                                                self.bins)
        x_graph,y_graph = screen_by_graph(x_hist,y_hist, rho_hist)
        x_polish,y_polish = polish_ends(x_graph,y_graph,self.tol_bins)
        self.x_fit,self.y_fit = fit_by_lowess(x_polish,y_polish,manual_frac)
        self.Predor = Predictor(self.x_fit,self.y_fit)

    def predict(self,x) -> np.ndarray:
        """
        predict Measured RT by Spectral library RT on the basis of Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT for predict
        
        Results:
            1-D np.ndarray Measured RT predict
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
            >>> model.fit(x,y)
            >>> model.predict(x)
        """
        return self.__predict(x)
        

    def __predict(self,x):
        x_pred_normal = self.nomral.normal("x",x)
        y_pred_normal = self.Predor.predict(x_pred_normal)
        y_pred = self.nomral.denormal("y",y_pred_normal)
        return y_pred