from scipy import signal
from scipy.integrate import trapezoid
from scipy.fft import fft,fftfreq
from scipy.signal import hann,correlate
from scipy.integrate import quad
import pandas as pd
import os
import numpy as np
import argparse
from statsmodels.tsa.stattools import acf

'''



'''
def get_FFTOfACF(acf: np.ndarray, fs=None):
    '''
        Performs the FFT of the autocorrelation function.
        Args
            acf: ensemble average of the ACF
            fs: sample rate to return a frequency vector
                None is returned if no sample rate is provided

        This FFT uses a Hanning windowing function to reduce
        spectral leakage. An amplitude correction factor is 
        used to correct for the windowing effects on the output
    '''

    print(f"PERFORMING FFT OF SIGNAL WITH SAMPLING RATE: {fs}")
    N = len(acf)
    wind = hann(N)
    X = acf*wind
    correction = np.sum(wind) / N

    fft_x = fft(X)
    if(fs is not None):
        freq = fftfreq(N,d=1/fs)
        freq = freq[:N//2]
    else:
        print("WARNING, no fs provided or could be inferred... This may cause erroneous spectra magnitudes")
        freq = fftfreq(N,d=1/1)

    fft_x = (2./fs)*np.abs(fft_x/correction)
    fft_x = fft_x[:N//2]

    return freq,fft_x


def modelIntegrand(kappa,k1,eps,L,eta,cL,ceta):
    A = (1-(k1**2)/(kappa**2)) / kappa
    C = 1.5
    p0 = 2
    beta = 5.2
    fL = ((kappa*L)/(((kappa*L)**2 + cL)**0.5))**(5./3.+p0)
    B = -1*beta*(((kappa*eta)**4 + ceta**4)**0.25 - ceta)
    feta = np.exp(B)

    mod = C*fL*feta*(eps**(2./3.))*(kappa**(-5./3.))

    return A*mod

def get_modelSpectra(tke:float,L:float,CL=6.78,Ceta=0.4,startKappa=0.1,endKappa=20000,N=15000):
    '''
        Generates a 1D model spectrum from provided tke and dissipation 
        parameters. CL and Ceta parameters can be tuned to improve
        the model agreement.
        Args
            tke: kinetic energy input
            eps: dissipation rate input
            CL/Ceta: Model parameters. Default parameters work well
                     for high Re flows. Tune to agree with tke and eps
            startKappa: starting wavenumber
            endKappa: final wavenumber value to integrate to
            N: how many points to include for output array

        Number of loop iterations is directly determined by N/endKappa
    '''


    print(f"GENERATING MODEL SPECTRUM. THIS MAY TAKE A WHILE")
    nu = 1.1818e-6   #assume typical viscosity of water (15 C)
    eps = (tke**(3/2)) / L
    eta = ((nu**3)/eps)**0.25

    E11_model = np.empty(N)
    k1 = np.logspace(np.log10(startKappa),np.log10(endKappa),N)
    idx = 0
    for i in k1:
        #print(k1,eps,L,eta,CL,Ceta)
        E11_model[idx] = quad(modelIntegrand,i,endKappa,args=(i,eps,L,eta,CL,Ceta))[0]
        idx = idx + 1
        

    integral_E11 = trapezoid(E11_model,x=k1)
    
    return k1,E11_model,integral_E11
def get_kolmogorovScaling(k1,E11,tke,eps):
    nu = 1.1818e-6
    eta = ((nu**3)/eps)**(0.25)
    k1eta = eta*k1

    kolm = E11*(eps*(nu**5))**(-0.25)

    return k1eta,kolm

def get_velStats(filename: str, sep, tpos=0,upos=1,vpos=2,wpos=3, header=0):
    '''
        Use to extract Reynolds stresses and turbulent kinetic energy
        from the time series file. 

        Args:
            filename: input file to read
            sep: delimiter being used
            tpos, upos, vpos, and wpos: specify which column position 
                (zero indexed) of time and velocity components. Order
                listed is the assume positions by default
    '''
        
    print(f'READING DATASET: {filename}')
    data = pd.read_csv(filename, sep=sep,header=header)
    col_names = data.columns.tolist()
    vel_components = []
    if(len(col_names) < 4):
        if(tpos >= 0 and upos >= 0 and vpos >= 0 and wpos >= 0):
            print("ERROR: not enough columns in data file for all three velocity components. Please use turn off missing components and refer to --help command for details")
            os._exit()
        else:
            for i in range(0,len(col_names)):
                if(i==upos or i==vpos or i==wpos):
                    vel_components.append(col_names[i])
    else:
        vel_components = [col_names[upos],col_names[vpos], col_names[wpos]]

    RSS = [None, None, None, None, None, None]
    print(f'GETTING MEAN VELOCITY STATISTICS OF: {filename}')
    for i in range(0,len(vel_components)):
        for j in range(i,len(vel_components)):
            u = vel_components[i]
            v = vel_components[j]
            mean_u = data[u].mean(axis=0)
            x = (data[u] - mean_u).to_numpy()
            sm = lambda a : np.mean(a**2)
            if(i == j):
                if(u == vel_components[0]):
                    if RSS[0] is None:
                        RSS[0] = sm(x)
                    else:
                        print("ERROR")
                elif(u == vel_components[1]):
                    if RSS[1] is None:
                        RSS[1] = sm(x)
                    else:
                        print("ERROR")
                elif(u == vel_components[2]):
                    if RSS[2] is None:
                        RSS[2] = sm(x)
                    else:
                        print("ERROR")
            else:
                mean_v = data[v].mean(axis=0)
                y = (data[v] - mean_v).to_numpy()
                if(u == vel_components[0]):
                    if(v == vel_components[1]):
                        if RSS[3] is None:
                            RSS[3] = np.mean(x*y)
                        else:
                            print("ERROR")
                    elif(v == vel_components[2]):
                        if RSS[4] is None:
                            RSS[4] = np.mean(x*y)
                        else:
                            print("ERROR")
                    else:
                        print("ERROR")
                elif(u == vel_components[1]):
                    if(v == vel_components[2]):
                        if RSS[5] is None:
                            RSS[5] = np.mean(x*y)
                        else:
                            print("ERROR")
                    else:
                        print("ERROR")
                else:
                    if(uv is None or uw is None or vw is None):
                        print("ERROR")


    if(len(vel_components) < 3):
        for index,r in enumerate(RSS):
            if r is None:
                if(index < 3 and index > 0):
                    RSS[index] = RSS[0] 
                else:
                    RSS[index] = 0.0

    tke = 0.5*(RSS[0]+RSS[1]+RSS[2])
    return RSS[0],RSS[1],RSS[2],RSS[3],RSS[4],RSS[5],tke

def get_parabolaCoeffs(x_data,y_data):
    '''
        Returns coefficients of a parabola provided
        Args:
            x_data: list of order x coordinates
            y_data: corresponding list of y coordiantes

        Use the 3 points centered at and around r/tau = 0 
        of the autocorrelation function to define an
        osculating parabola for autocorrelation at f(r=0)
    '''
    
    return np.polyfit(x_data,y_data,2)

def parabola(x,a,b,c):
    '''
        use to return a y-value from the 3 coefficient values
        defining a parabola and a given an x value. 
        Args:
            x: independent data point value to evaluate parabola
            a,b,c: parabola coefficients

        This is used output osculating parabola with the
        autocorrelation function at f(r=0) for an estimate
        of the Taylor microscale
    '''
    return a*x**2 + b*x + c

def get_autocorr(filename: str, var: str, fft=False, time=None, fs=None, sep=' ',header=0):
    '''
        This function performs the autocorrelation given a 
        timeseries of a specified variable.
        Args:
            filename: data file name to process
            var: variable name or index to perform the autocorr on
            fft: bool switch for using direct or convolution method
            time: index or var name of the time column for calculating time lags
            fs: alternate way to calculate time lags using sampling rate
            sep: delimiter option
    '''

    data = pd.read_csv(filename, sep=sep, header =header)
    col_names = data.columns.tolist()
    if(isinstance(var,int)):
        ivar = col_names[var]
    else:
        ivar = var
    print(f"PERFORMING AUTCORRELATION OF VARIABLE: {ivar}" )
    mean_val = data[ivar].mean(axis=0)
    x = (data[ivar] - mean_val).to_numpy()
    #f_r = acf(x, fft=fft, nlags=len(x))
    f_r = correlate(x,x,mode='full',method='direct')
    f_r = f_r[len(f_r)//2:-1]
    f_r = f_r/f_r[0]

    tau = None
    if(isinstance(time,int) and time < 0):
        time = None
    if(time is not None):
        if(isinstance(time,int)):
            itime = col_names[time]
        else:
            itime = time
        t = data[itime].to_numpy()
        tau = np.linspace(0,len(f_r)*(t[1]-t[0]), len(f_r))
    elif(fs is not None):
        tau = np.linspace(0,(len(f_r)-1)/fs, len(f_r))
    else:
        print("ERROR: no time variable was specified and no sampling rate was give either")
        os._exit()

    f_r = f_r.reshape((-1,1))
    return tau,f_r,mean_val


def write_autocorr(filename: str, tau, fr, coeffs, tau_end=10, n= 50):
    ''' 
        Writes out the autocorrelation function using tecplot ascii point format.
        This is equvilant to a deliminted text file if the file header and zone 
        headers are ignored.
        Args
            filename: output file name
            tau: array of time lags
            fr: autocorrelation at the time lags
            tau_end: final lag vale (index of tau) for plotting the osculating parabola
            n: how many data points to use for parabola plot

        The osculating parabola is also outputed up to a specified lag and n points
        (50 by deffault).
    '''
    if(tau_end > (len(tau)-1)):
        tau_end = len(tau) - 1 
    x=np.linspace(0,tau[tau_end],n)
    y=np.empty(n)
    analytic=np.empty(n)
    a,b,c = coeffs
    ms = np.roots(coeffs)[0]
    for i in range(0,n):
        y[i] = parabola(x[i],a,b,c)
        analytic[i] = np.exp(-1*(x[i]**2)/ms**2)

    print("STATUS: Writing out ensemble averaged autocorrelation function: \"", filename, "\"")
    with open(filename, 'w') as of:
        of.write("TITLE = \"Autocorrelation Function\"\n")
        of.write(f"VARIABLES = \"tau\", \"f(r)\", \"parabola(n_3)\",\"analytic\"\n")
        of.write(f"ZONE T = \"Ensemble Avg. Autocorrelation\" I = {len(tau)}, PASSIVEVARLIST=[3,4]\n")
        for i in range(0,len(tau)): 
           of.write(f"{tau[i]} {fr[i]}\n") 
        of.write(f"ZONE T = \"Parbolic Fit of First 3 Points\" I = {n}, PASSIVEVARLIST=[2]\n")
        for i in range(0,n): 
           of.write(f"{x[i]} {y[i]} {analytic[i]}\n") 
        of.close()

def write_spectra(filename: str, f: np.ndarray, Y: np.ndarray, fname: str, Yname: str):
    '''
        Writes out a tecplot ascii point formatted data filea
        of a spectrum in frequency/wavenumber domain. This can 
        be used multiple times to output both FFTs of the 
        velocity, FFT of the autocorrelation, E11, or E11 with 
        Kolmogorov scalings
        Args
            filename: output file name
            f: frequency/wavenumber array
            Y: spectrum amplitude array
            Yname: name of spectrum var for output file
            fname: name of freq/wavenumber var for output file

        
    '''

    print(f"STATUS: Writing out ensemble averaged {Yname} to \"{filename}\"")
    with open(filename, 'w') as of:
        of.write(f"TITLE = \"{Yname} Spectrum\"\n")
        of.write(f"VARIABLES = \"{fname}\", \"{Yname}\"\n")
        of.write(f"ZONE T = \"Ensemble Avg. {Yname}\" I = {len(f)}\n")
        for i in range(0,len(f)): 
           of.write(f"{f[i]} {Y[i]}\n") 
        of.close()

def write_scales2Excel(Ubar, uu, k, l0, microscale, macroscale,fname):
    '''
        writes an excel tables for both the micro and macro scales
        provided the mean axial core velocity, longitudinal RSS,
        vortex length scale (l0), and Taylor micro and macro scales
        Args
            Ubar: mean core velocity
            uu: mean uu Reynolds stress (longitudinal direction of vortex)
            k: tke value
            l0: vortex width characteristic length scale
            micro/macroscale: Taylor micro and macroscale
            fname: output excel filename
    '''

    nu = 1.1818e-6   #assume typical viscosity of water (15 C)

    fU = '<U>'
    fuu = f'<u\N{SUPERSCRIPT TWO}>'
    fk = 'k'
    fu0 = 'u\N{SUBSCRIPT ZERO}'
    fup = 'u\''
    fl0 = 'l\N{SUBSCRIPT ZERO}'
    fL = 'L'
    feps = '\N{GREEK SMALL LETTER EPSILON}'
    fReL = 'ReL'
    fBMLambdaf = 'BM \u03bbf' 
    fBMReLambda = 'BM Re\u03bb'
    fReLambda = 'Re\u03bb'
    fBMeta = '\N{GREEK SMALL LETTER ETA}'
    feta = '\N{GREEK SMALL LETTER ETA}'
    ftauE = '\u03c4E'
    fT = 'T'
    flambdaf = '\u03bbf'
    fclambdaf = '\u039Bf'
    fnu = '\N{GREEK SMALL LETTER NU}'
    
    #SHEET 1 - MACROSCALES
    u0 = np.sqrt(k)
    uprime = np.sqrt(2/3*k)
    #uprime = np.sqrt(uu)
    L = l0 # / 0.43 
    eps = (k**(3/2)) / L
    ReL = u0*L / (nu)
    BM_microscale = np.sqrt(20)*L / np.sqrt(ReL)
    BM_Rlambda = uprime*BM_microscale / (np.sqrt(2)*nu)
    BM_eta = ((nu**3) / eps)**(0.25)

    sheet1_row_names = [fU, fuu, fk, fup, fu0, fl0, fL, feps, fReL, fBMLambdaf, fBMReLambda, fBMeta] 
    sheet1_data = { 'Value' : [Ubar, uu, k, uprime, u0, l0, L, eps, ReL, BM_microscale, BM_Rlambda,BM_eta] , 'Units' : ['m/s', 'm^2/s^2', 'm^2/s^2', 'm/s', 'm/s', 'm' , 'm', 'm^2/s^3', '-', 'm', '-', 'm'], 
                    'Method/Notes' : ['Mean', 'Mean Square', 'TKE', 'sqrt(tke)', 'sqrt(2/3k)', 'vortex width', 'same as vortex width', 'dissipation', 'Reynolds number u0*L/nu', 'Benchmark microscale', 
                    'Benchmark lambda Re', 'Benchmark Kolmogorov length scale' ]}

    sheet1_df = pd.DataFrame(sheet1_data, index=sheet1_row_names)


    #SHEET 2 - MICROSCALES

    Re_lambda = microscale / np.sqrt(2) * uprime / nu
    eps_micro = 30 / ((Ubar*microscale)**2) * (uprime**2) * nu
    eta = ((nu**3) / eps_micro)**(0.25)
    
    sheet2_row_names = [ftauE,fT,flambdaf, fclambdaf, fReLambda, feps, feta] 
    sheet2_data = { 'Value' : [microscale, macroscale,Ubar*microscale, Ubar*macroscale, Re_lambda, eps_micro, eta] , 'Units' : ['s','s','m', 'm', '-', 'm^2/s^3', 'm'],
                    'Method/Notes' : ['Parabola Root', 'Trapezoidal Integration','Taylor Frozen Turbulence Hypothesis', 'Taylor Frozen Turbulence Hypothesis', 'Microscale Re', 
                    'Dissipation based on microscales', 'Kolmogorov length scale' ]}
    
    sheet2_df = pd.DataFrame(sheet2_data, index=sheet2_row_names)

    #SHEET 3 - STATISTICS

    with pd.ExcelWriter(fname, engine='openpyxl') as exWriter:
        sheet1_df.to_excel(exWriter, sheet_name='Macroscales')
        sheet2_df.to_excel(exWriter, sheet_name='Microscales')

def write_rss2excel(avg_rss,fname):
    '''
        Writes out table of RSS values to excel file
        Args
            avg_rss: 2D array of the ensemble averaged RSS components and tke
            fname: output excel filename
    '''
    
    rss_row_names = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'tke'] 
    rss_data = { 'Value' : avg_rss, 'Units' : ['m^2/s^2','m^2/s^2','m^2/s^2','m^2/s^2','m^2/s^2','m^2/s^2','m^2/s^2',]}

    rss_df = pd.DataFrame(rss_data, index=rss_row_names)
    
    with pd.ExcelWriter(fname, engine= 'openpyxl', mode='a') as exWriter:
        rss_df.to_excel(exWriter, sheet_name='RSS')

def write_reportExcel(U, avg_rss, l0, microscale, macroscale, cf):
    write_scales2Excel(U,avg_rss[0],avg_rss[-1],l0,microscale,macroscale, cf)
    write_rss2excel(avg_rss, cf)
    

def process_ensembleData(acfs: np.ndarray, tau: np.ndarray, Ubar: float, rss: float, l0: float, cf):
    '''
        Processes the ensemble of data results if there are multiple realizations/files
        of the same case

        Args:
            acfs: 2D array (shape: lags x realizations)
            tau: 1D array (lags should be consistent between all realizations)
            Ubar: 1D array (number of realizatons)
            rss: 2D array (rss component x realizations)
            l0: float (nominal vortex width should be the same for all realizations)
            cf: current folder used for saving excel report in a specified data folder
    '''

    avg_acf = acfs.mean(axis=1)
    avg_U = Ubar.mean()
    avg_rss = rss.mean(axis=1)

    
    freq, RE = get_FFTOfACF(avg_acf, fs=1/(tau[1]-tau[0]))
    fname = cf+"/ensemble_RE.dat"
    write_spectra(fname,freq,RE,'freq (Hz)', 'RE (s)')
    fname = cf+"/ensemble_E11hat.dat"
    E11hat = RE*2.*avg_rss[0]
    write_spectra(fname,freq,E11hat,'freq (Hz)', 'E11_hat (m^2/s)')
    fname = cf+"/ensemble_E11.dat"
    wavenumber = freq*2.*np.pi / avg_U
    E11 = E11hat*avg_U / (2.*np.pi)
    write_spectra(fname,wavenumber,E11,'kappa (1/m)', 'E11 (m^3/s^2)')

    # Fitting a parabola around f(r=0), 3pt fit (function is symmetric)
    coeffs = get_parabolaCoeffs(np.array([-1*tau[1], tau[0], tau[1]]), np.array([avg_acf[1],avg_acf[0],avg_acf[1]]))
    fname = cf+"/ensemble_autocorrelation.dat"
    write_autocorr(fname, tau=tau, fr=avg_acf, coeffs=coeffs, n=75,tau_end=40) 

    microscale = np.roots(coeffs)[0]
    zerocrossing = np.where(np.diff(np.sign(avg_acf)))[0][0]  #get first zero crossing index
    
    #trapezoidal integration being used to get L11 macroscale from acf
    macroscale = trapezoid(avg_acf[0:zerocrossing], tau[0:zerocrossing])

    #calculate remaining area undercurve via interpolation and triangular area formula
    slope = (avg_acf[zerocrossing+1] - avg_acf[zerocrossing])/(tau[zerocrossing+1]-tau[zerocrossing])
    area = -0.5*(avg_acf[zerocrossing]**2)/slope
    macroscale=macroscale+area

    excelName = cf+"/ensemble_turbulence_table_report.xlsx"
    write_reportExcel(avg_U, avg_rss, l0, microscale, macroscale,excelName)

    k1,E11_model,intE11 = get_modelSpectra(avg_rss[-1],l0)
    print(f"\tOutputting the integral of E11 over uu: {intE11/avg_rss[0]}")
    fname = cf+"/ensemble_modelE11.dat"
    write_spectra(fname,k1,E11_model,'k1 (1/m)', 'Model E11 (m^3/s^2)')

    epsMac = (avg_rss[6]**(3/2)) / l0 #macroscale dissipation
    epsMic = 30 / ((avg_U*microscale)**2) * (avg_rss[0])*(1.1818e-6) 
    k1eta, macKolmE11 = get_kolmogorovScaling(wavenumber,E11,avg_rss[6],epsMac)
    k1eta, micKolmE11 = get_kolmogorovScaling(wavenumber,E11,avg_rss[6],epsMic)
    k1etaMod, macKolmModE11 = get_kolmogorovScaling(k1,E11_model,avg_rss[6],epsMac)
    k1etaMod, micKolmModE11 = get_kolmogorovScaling(k1,E11_model,avg_rss[6],epsMic)
    
    fname = cf+"/ensemble_macro_kolmogorov.dat"
    write_spectra(fname,k1eta,macKolmE11,'k1eta', 'E11*(nu^5*eps)^-1/4')
    fname = cf+"/ensemble_micro_kolmogorov.dat"
    write_spectra(fname,k1eta,micKolmE11,'k1eta', 'E11*(nu^5*eps)^-1/4')

    fname = cf+"/ensemble_macro_modelKolmogorov.dat"
    write_spectra(fname,k1etaMod,macKolmModE11,'k1eta', 'E11*(nu^5*eps)^-1/4')
    fname = cf+"/ensemble_micro_modelKolmogorov.dat"
    write_spectra(fname,k1etaMod,micKolmModE11,'k1eta', 'E11*(nu^5*eps)^-1/4')

def main(files, Us: float, Ls: float, Ts: float, l0: float, fs: float, delim, header_line, col_info, provided_RSS=None): 
    '''
        Main program execution
        Args
            files: list of files to include for ensemble analysis
            Us: Velocity scaling to use
            Ls: Length scaling to use 
            l0: characteristic eddy length scale (i.e. nominal vortex width)
    '''
    print("MAIN() CALLED")

    current_count = 0
    current_folder = None
    acfs = None
    meanV = None
    meanRSS = None
    tau = None

    l0 = l0*Ls

    for f in files:
        head, tail = os.path.split(f)

        if(current_folder != None and current_folder != head):
            #the next file is in different folder, 
            #perform ensemble average analysis from the previous folder files

            print("STATUS: Finished analysis for all files in directiory \"", current_folder,"\"")
            print("Number of datafiles used = ", current_count)
            process_ensembleData(acfs, tau, meanV, meanRSS,l0, current_folder)

            #reset ensemble variables to be used for the next folder
            acfs = None
            meanV = None
            meanRSS = None
            current_count = 0
            current_folder = head
        
        if(current_folder == None):
            current_folder = head

        tau, f_r, Ubar = get_autocorr(f, col_info[1], time=col_info[0], sep=delim,fs=fs) 
        #if(tau is None):
        #    print(f"TIME VARIABLE NOT PROVIDED...Deducing time lags using sampling rate")
        #    tau = np.linspace(0,fs*len(f_r),len(f_r))
        tau = tau*Ts
        

        Ubar = Ubar*Us
        if(provided_RSS is None):
            uu,vv,ww,uv,uw,vw,tke = get_velStats(f,delim,tpos=col_info[0],upos=col_info[1],vpos=col_info[2],wpos=col_info[3])
            uu = uu*Us*Us
            vv = vv*Us*Us
            ww = ww*Us*Us
            uv = uv*Us*Us
            uw = uw*Us*Us
            vw = vw*Us*Us
            tke =tke*Us*Us
        else:
            uu,_ = get_velStats(f,delim,tpos=col_info[0],upos=col_info[1],vpos=col_info[2],wpos=col_info[3])
            uu = uu*Us*Us
            vv = provided_RSS[0]*Us*Us
            ww = provided_RSS[1]*Us*Us
            uv = provided_RSS[2]*Us*Us
            uw = provided_RSS[3]*Us*Us
            vw = provided_RSS[4]*Us*Us
            tke =(0.5*(uu+vv+ww))*Us*Us
        #add current autocorrelation to the ensemble
        if acfs is None:
            acfs = f_r
            meanV = np.empty(1)
            meanV[0] = Ubar
            meanRSS = np.empty((7,1))
            meanRSS[0] = uu
            meanRSS[1] = vv
            meanRSS[2] = ww
            meanRSS[3] = uv
            meanRSS[4] = uw
            meanRSS[5] = vw
            meanRSS[6] = tke
        else:
            if(acfs.shape[0] != f_r.shape[0]):
                #lengths are not the same
                if (f_r.shape[0] < acfs.shape[0]):
                    #trim previous autocorrelations
                    newSize = f_r.shape[0]
                    acfs = np.take(acfs, indices=range(newSize), axis=0)
                    print("TRIMMING TO MATCH SIZE OF FILE: ", tail)
                else:
                    newSize = acfs.shape[0]
                    f_r = f_r[0:newSize]
                    tau = tau[0:newSize]

            acfs = np.concatenate((acfs,f_r), axis=1)
            meanV = np.concatenate((meanV,np.array([Ubar])), axis=0)
            newRSS = np.array([uu,vv,ww,uv,uw,vw,tke]).reshape(7,1)
            meanRSS = np.concatenate((meanRSS,newRSS),axis=1)

        print("ARRAY SHAPE OF AUTOCORRECTON: ", acfs.shape)

        current_count = current_count + 1
        ##END OF FILE ITERATION LOOP

    if(current_folder != None):
        print("STATUS: Finished analysis for all files in directiory \"", current_folder,"\"")
        print("\tNumber of datafiles used = ", current_count)
        
        process_ensembleData(acfs, tau, meanV, meanRSS,l0,current_folder)


if __name__ == "__main__":
    import argparse
    parser =  argparse.ArgumentParser(description="Turbulence analysis and statistics of a velocity timeseries: A program to determine the Reynolds shear stresses, longitudinal (i.e. Vx) velocity"
                                                  " autocorrelation function, macro and micro turbulence scales/parameters, and the longitudinal 1D energy specta plots from velocity time history data.")

    parser.add_argument('--files', nargs='*', help='filenames of the velocity timeseries with Time, Vx, Vy, Vz variables')

    parser.add_argument('--l0', type=float, help='Reference eddy length scale (i.e. vortex width) in m')
    parser.add_argument('--Us', type=float, default=1.0, help='If velocities are nondimensional, provide reference velocity scale (i.e. Carriage Speed) in m/s. Us = 1 (i.e. dimensional) by default')
    parser.add_argument('--Ls', type=float, default=1.0, help='If length scales are nondimensional, provide reference length scale (i.e. model length) in m. Ls = 1 (i.e. dimensional) by default')
    parser.add_argument('--Ts', type=float, default=1.0, help='If time values are nondimensional, provide reference length scale (i.e. model length) in m. Ts = 1 (i.e. dimensional) by default')
    parser.add_argument('--fs', type=float, default=1.0, help='A sampling rate if no time column is included in data files')

    parser.add_argument('--delim', type=str, default=' ', help='Delimiter used for data file. Default is a "space" character.')
    parser.add_argument('--vars', nargs=4,type=int, help='Variable ordering is assumed to be [Time, Vx, Vy, Vz]. If this is not true, provide the column number (with zero-based indexing) of the file'
                                                         ' variable supplied in the same order listed above. Use -1 for variables not included.', default=[0,1,2,3],
                                                         metavar=('Time Col#', 'Vx Col#' ,'Vy Col#', 'Vz Col#'))
    parser.add_argument('--header',type=int,default=0, help='Line number (zero indexed) of the file header (i.e. line containing column/var names. Set to -1 if no header is included and set variable.'
                                                            ' information based on -v/--vars flag, -c/--cols flag, or default variable list provided by the --vars flag.')

    parser.add_argument('--RSS',nargs=5,help='By default, this code calculates and outputs the RSS values when provided data for each velocity component. If it is desired to only use this code'
                                               ' to generate the 1D (i.e. U velocity) energy spectra and macro/micro scale results, you can provide missing RSS results so accurate model spectra'
                                               ' RSS table values will be outputted. Isotropic conditions are assumed otherwise',
                                               metavar=('vv', 'ww', 'uv', 'uw', 'vw'))


    args = parser.parse_args()

    main(args.files, args.Us, args.Ls, args.Ts, args.l0, args.fs, args.delim, args.header, args.vars, args.RSS)
        
