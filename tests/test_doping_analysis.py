"""
Doping Analysis Module for High-Temperature Superconductors

This module provides tools for analyzing doping effects in HTSC materials,
including carrier concentration calculations, optimal doping determination,
and comparative analysis between different doping mechanisms.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class DopingAnalysis:
    """
    Main class for analyzing doping effects in high-temperature superconductors.
    
    Supports analysis of hole-doped cuprates, electron-doped cuprates,
    and substitutionally doped iron-based superconductors.
    """
    
    def __init__(self, material_type: str, doping_type: str):
        """
        Initialize the DopingAnalysis object.
        
        Parameters:
        -----------
        material_type : str
            Type of material ('cuprate', 'pnictide', 'chalcogenide')
        doping_type : str
            Type of doping ('hole', 'electron', 'substitutional')
        """
        self.material_type = material_type
        self.doping_type = doping_type
        self.data = None
        
        # Physical constants
        self.elementary_charge = 1.602e-19  # C
        self.hall_constant_factor = 1/(self.elementary_charge)
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load experimental data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV data file
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        try:
            self.data = pd.read_csv(filepath)
            return self.data
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return None
    
    def calculate_carrier_concentration(self, hall_coefficient: np.ndarray) -> np.ndarray:
        """
        Calculate carrier concentration from Hall coefficient data.
        
        Parameters:
        -----------
        hall_coefficient : np.ndarray
            Hall coefficient values in cm³/C
            
        Returns:
        --------
        np.ndarray
            Carrier concentration in cm⁻³
        """
        # n = 1/(e * R_H) where R_H is Hall coefficient
        carrier_concentration = 1 / (self.elementary_charge * np.abs(hall_coefficient))
        return carrier_concentration
    
    def determine_doping_level(self, composition: str, dopant_concentration: float) -> float:
        """
        Determine effective doping level from chemical composition.
        
        Parameters:
        -----------
        composition : str
            Chemical formula of the compound
        dopant_concentration : float
            Concentration of dopant atoms
            
        Returns:
        --------
        float
            Effective doping level (holes or electrons per formula unit)
        """
        if self.material_type == 'cuprate':
            # For cuprates, doping level often relates to holes/electrons per Cu
            if 'YBCO' in composition.upper():
                # YBa₂Cu₃O₇₋δ: oxygen deficiency creates holes
                return dopant_concentration * 2  # Approximate conversion
            elif 'LSCO' in composition.upper():
                # La₂₋ₓSrₓCuO₄: Sr²⁺ replaces La³⁺, creating holes
                return dopant_concentration
            elif 'NCCO' in composition.upper():
                # Nd₂₋ₓCeₓCuO₄: Ce⁴⁺ replaces Nd³⁺, creating electrons
                return dopant_concentration
                
        elif self.material_type == 'pnictide':
            # For iron pnictides, doping affects Fe sites
            return dopant_concentration
            
        return dopant_concentration
    
    def analyze_optimal_doping(self, tc_data: np.ndarray, 
                             doping_data: np.ndarray) -> Dict[str, float]:
        """
        Find optimal doping level for maximum Tc.
        
        Parameters:
        -----------
        tc_data : np.ndarray
            Critical temperature values
        doping_data : np.ndarray
            Corresponding doping levels
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing optimal doping level and maximum Tc
        """
        # Find maximum Tc and corresponding doping level
        max_tc_idx = np.argmax(tc_data)
        optimal_doping = doping_data[max_tc_idx]
        max_tc = tc_data[max_tc_idx]
        
        # Fit parabolic function around the maximum for better precision
        if len(tc_data) > 3:
            # Use points around the maximum for fitting
            fit_range = max(3, len(tc_data) // 4)
            start_idx = max(0, max_tc_idx - fit_range)
            end_idx = min(len(tc_data), max_tc_idx + fit_range + 1)
            
            x_fit = doping_data[start_idx:end_idx]
            y_fit = tc_data[start_idx:end_idx]
            
            # Fit parabolic function: Tc = a(x - x0)² + Tc_max
            def parabola(x, a, x0, tc_max):
                return a * (x - x0)**2 + tc_max
            
            try:
                popt, _ = curve_fit(parabola, x_fit, y_fit, 
                                  p0=[-100, optimal_doping, max_tc])
                refined_optimal_doping = popt[1]
                refined_max_tc = popt[2]
                
                return {
                    'optimal_doping': refined_optimal_doping,
                    'max_tc': refined_max_tc,
                    'fit_parameters': popt
                }
            except:
                pass
        
        return {
            'optimal_doping': optimal_doping,
            'max_tc': max_tc,
            'fit_parameters': None
        }
    
    def calculate_superconducting_dome_width(self, tc_data: np.ndarray,
                                           doping_data: np.ndarray,
                                           threshold: float = 5.0) -> Dict[str, float]:
        """
        Calculate the width of the superconducting dome.
        
        Parameters:
        -----------
        tc_data : np.ndarray
            Critical temperature values
        doping_data : np.ndarray
            Corresponding doping levels
        threshold : float
            Minimum Tc to consider as superconducting
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing dome width and boundaries
        """
        # Find indices where Tc > threshold
        sc_indices = np.where(tc_data > threshold)[0]
        
        if len(sc_indices) == 0:
            return {'width': 0, 'lower_bound': 0, 'upper_bound': 0}
        
        lower_bound = doping_data[sc_indices[0]]
        upper_bound = doping_data[sc_indices[-1]]
        width = upper_bound - lower_bound
        
        return {
            'width': width,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def compare_doping_types(self, data1: pd.DataFrame, data2: pd.DataFrame,
                           label1: str, label2: str) -> Dict[str, any]:
        """
        Compare two different doping types or materials.
        
        Parameters:
        -----------
        data1, data2 : pd.DataFrame
            Data for comparison
        label1, label2 : str
            Labels for the datasets
            
        Returns:
        --------
        Dict[str, any]
            Comparison results
        """
        comparison = {}
        
        # Compare maximum Tc
        max_tc1 = data1['tc_k'].max()
        max_tc2 = data2['tc_k'].max()
        
        # Compare optimal doping
        optimal1 = self.analyze_optimal_doping(data1['tc_k'].values, 
                                             data1['doping_level'].values)
        optimal2 = self.analyze_optimal_doping(data2['tc_k'].values,
                                             data2['doping_level'].values)
        
        # Compare dome widths
        dome1 = self.calculate_superconducting_dome_width(data1['tc_k'].values,
                                                        data1['doping_level'].values)
        dome2 = self.calculate_superconducting_dome_width(data2['tc_k'].values,
                                                        data2['doping_level'].values)
        
        comparison = {
            'max_tc': {label1: max_tc1, label2: max_tc2},
            'optimal_doping': {label1: optimal1['optimal_doping'], 
                             label2: optimal2['optimal_doping']},
            'dome_width': {label1: dome1['width'], label2: dome2['width']},
            'tc_asymmetry': abs(max_tc1 - max_tc2) / max(max_tc1, max_tc2)
        }
        
        return comparison
    
    def analyze_transport_properties(self, temperature: np.ndarray,
                                   resistivity: np.ndarray,
                                   hall_coefficient: np.ndarray) -> Dict[str, any]:
        """
        Analyze transport properties as a function of temperature and doping.
        
        Parameters:
        -----------
        temperature : np.ndarray
            Temperature values
        resistivity : np.ndarray
            Resistivity values
        hall_coefficient : np.ndarray
            Hall coefficient values
            
        Returns:
        --------
        Dict[str, any]
            Transport analysis results
        """
        analysis = {}
        
        # Calculate carrier concentration
        carrier_conc = self.calculate_carrier_concentration(hall_coefficient)
        analysis['carrier_concentration'] = carrier_conc
        
        # Calculate conductivity
        conductivity = 1 / resistivity
        analysis['conductivity'] = conductivity
        
        # Calculate mobility (σ = n * e * μ)
        mobility = conductivity / (carrier_conc * self.elementary_charge)
        analysis['mobility'] = mobility
        
        # Analyze temperature dependence
        if len(temperature) > 1:
            # Linear fit for high-temperature region
            high_temp_mask = temperature > 200  # K
            if np.sum(high_temp_mask) > 2:
                coeffs = np.polyfit(temperature[high_temp_mask], 
                                  resistivity[high_temp_mask], 1)
                analysis['resistivity_slope'] = coeffs[0]
                analysis['residual_resistivity'] = coeffs[1]
        
        return analysis
    
    def calculate_pseudogap_temperature(self, thermopower: np.ndarray,
                                      temperature: np.ndarray) -> float:
        """
        Estimate pseudogap temperature from thermopower measurements.
        
        Parameters:
        -----------
        thermopower : np.ndarray
            Thermopower values
        temperature : np.ndarray
            Temperature values
            
        Returns:
        --------
        float
            Estimated pseudogap temperature
        """
        # Find temperature where thermopower changes sign or has maximum slope
        if len(thermopower) < 3:
            return 0
        
        # Calculate derivative
        dt_ds = np.gradient(thermopower, temperature)
        
        # Find maximum slope (most negative derivative)
        max_slope_idx = np.argmin(dt_ds)
        
        return temperature[max_slope_idx]
    
    def fit_tc_vs_doping(self, doping_data: np.ndarray, 
                        tc_data: np.ndarray) -> Dict[str, any]:
        """
        Fit various models to Tc vs doping relationship.
        
        Parameters:
        -----------
        doping_data : np.ndarray
            Doping levels
        tc_data : np.ndarray
            Critical temperatures
            
        Returns:
        --------
        Dict[str, any]
            Fitting results for different models
        """
        fits = {}
        
        # Parabolic fit (dome-like)
        def parabolic(x, a, x0, tc_max):
            return np.maximum(0, a * (x - x0)**2 + tc_max)
        
        try:
            popt_para, pcov_para = curve_fit(parabolic, doping_data, tc_data,
                                           p0=[-100, 0.16, 100])
            fits['parabolic'] = {
                'parameters': popt_para,
                'covariance': pcov_para,
                'r_squared': self._calculate_r_squared(tc_data, 
                                                     parabolic(doping_data, *popt_para))
            }
        except:
            fits['parabolic'] = None
        
        # Abrikosov-Gor'kov fit for pair breaking
        def abrikosov_gorkov(x, tc0, x_opt, alpha):
            return tc0 * (1 - alpha * (x - x_opt)**2)
        
        try:
            popt_ag, pcov_ag = curve_fit(abrikosov_gorkov, doping_data, tc_data,
                                       p0=[100, 0.16, 10])
            fits['abrikosov_gorkov'] = {
                'parameters': popt_ag,
                'covariance': pcov_ag,
                'r_squared': self._calculate_r_squared(tc_data,
                                                     abrik