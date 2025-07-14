"""
Transport Properties Module for High-Temperature Superconductors

This module provides tools for analyzing transport properties of HTSC materials,
including resistivity, Hall effect, thermopower, and thermal conductivity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class TransportProperties:
    """
    Class for analyzing transport properties of HTSC materials.
    """
    
    def __init__(self, material_type: str = 'cuprate'):
        """
        Initialize TransportProperties object.
        
        Parameters:
        -----------
        material_type : str
            Type of material ('cuprate', 'pnictide', 'chalcogenide')
        """
        self.material_type = material_type
        
        # Physical constants
        self.k_B = 1.381e-23  # J/K
        self.e = 1.602e-19    # C
        self.hbar = 1.055e-34 # J·s
        self.m_e = 9.109e-31  # kg
    
    def analyze_resistivity(self, temperature: np.ndarray,
                           resistivity: np.ndarray,
                           doping_level: float = None) -> Dict[str, float]:
        """
        Analyze resistivity data and extract transport parameters.
        
        Parameters:
        -----------
        temperature : np.ndarray
            Temperature values (K)
        resistivity : np.ndarray
            Resistivity values (Ω·cm)
        doping_level : float, optional
            Doping level for the sample
            
        Returns:
        --------
        Dict[str, float]
            Resistivity analysis results
        """
        results = {}
        
        # Residual resistivity (T → 0 limit)
        if len(temperature) > 5:
            low_temp_mask = temperature < 50
            if np.sum(low_temp_mask) > 2:
                # Linear extrapolation to T = 0
                coeffs = np.polyfit(temperature[low_temp_mask], 
                                  resistivity[low_temp_mask], 1)
                results['residual_resistivity'] = coeffs[1]  # Ω·cm
                results['temperature_coefficient'] = coeffs[0]  # Ω·cm/K
        
        # Room temperature resistivity
        if 300 in temperature or np.any(np.abs(temperature - 300) < 10):
            idx_300 = np.argmin(np.abs(temperature - 300))
            results['resistivity_300K'] = resistivity[idx_300]
        
        # Resistivity ratio (RRR)
        if 'residual_resistivity' in results and 'resistivity_300K' in results:
            results['resistivity_ratio'] = results['resistivity_300K'] / results['residual_resistivity']
        
        # Analyze temperature dependence
        results.update(self._analyze_temperature_dependence(temperature, resistivity))
        
        return results
    
    def _analyze_temperature_dependence(self, temperature: np.ndarray,
                                      resistivity: np.ndarray) -> Dict[str, float]:
        """
        Analyze temperature dependence of resistivity.
        """
        results = {}
        
        # High temperature region (T > 200K) - often linear
        high_temp_mask = temperature > 200
        if np.sum(high_temp_mask) > 3:
            T_high = temperature[high_temp_mask]
            rho_high = resistivity[high_temp_mask]
            
            # Linear fit: ρ = ρ₀ + αT
            coeffs = np.polyfit(T_high, rho_high, 1)
            results['high_temp_slope'] = coeffs[0]  # Ω·cm/K
            results['high_temp_intercept'] = coeffs[1]  # Ω·cm
        
        # Check for T² behavior (Fermi liquid)
        mid_temp_mask = (temperature > 50) & (temperature < 150)
        if np.sum(mid_temp_mask) > 5:
            T_mid = temperature[mid_temp_mask]
            rho_mid = resistivity[mid_temp_mask]
            
            # Fit ρ = ρ₀ + AT²
            def t_squared(T, rho0, A):
                return rho0 + A * T**2
            
            try:
                popt, _ = curve_fit(t_squared, T_mid, rho_mid)
                results['fermi_liquid_A'] = popt[1]  # Ω·cm/K²
                results['fermi_liquid_rho0'] = popt[0]  # Ω·cm
            except:
                pass
        
        return results
    
    def analyze_hall_effect(self, temperature: np.ndarray,
                           hall_coefficient: np.ndarray,
                           resistivity: np.ndarray) -> Dict[str, any]:
        """
        Analyze Hall effect data.
        
        Parameters:
        -----------
        temperature : np.ndarray
            Temperature values (K)
        hall_coefficient : np.ndarray
            Hall coefficient values (cm³/C)
        resistivity : np.ndarray
            Resistivity values (Ω·cm)
            
        Returns:
        --------
        Dict[str, any]
            Hall effect analysis results
        """
        results = {}
        
        # Calculate carrier concentration
        carrier_concentration = 1 / (self.e * np.abs(hall_coefficient))
        results['carrier_concentration'] = carrier_concentration
        
        # Determine carrier type
        results['carrier_type'] = 'holes' if np.mean(hall_coefficient) > 0 else 'electrons'
        
        # Calculate Hall mobility
        hall_mobility = np.abs(hall_coefficient) / (resistivity * 1e-2)  # cm²/V·s
        results['hall_mobility'] = hall_mobility
        
        # Calculate conductivity
        conductivity = 1 / (resistivity * 1e-2)  # S/cm
        results['conductivity'] = conductivity
        
        # Room temperature values
        if 300 in temperature or np.any(np.abs(temperature - 300) < 10):
            idx_300 = np.argmin(np.abs(temperature - 300))
            results['hall_coefficient_300K'] = hall_coefficient[idx_300]
            results['carrier_concentration_300K'] = carrier_concentration[idx_300]
            results['hall_mobility_300K'] = hall_mobility[idx_300]
        
        # Temperature dependence analysis
        results.update(self._analyze_hall_temperature_dependence(temperature, hall_coefficient))
        
        return results
    
    def _analyze_hall_temperature_dependence(self, temperature: np.ndarray,
                                           hall_coefficient: np.ndarray) -> Dict[str, float]:
        """
        Analyze temperature dependence of Hall coefficient.
        """
        results = {}
        
        # Check for sign change (multiband effects)
        if np.any(hall_coefficient > 0) and np.any(hall_coefficient < 0):
            results['sign_change'] = True
            # Find temperature of sign change
            sign_changes = np.where(np.diff(np.sign(hall_coefficient)))[0]
            if len(sign_changes) > 0:
                results['sign_change_temperature'] = temperature[sign_changes[0]]
        else:
            results['sign_change'] = False
        
        # Calculate Hall coefficient slope
        if len(temperature) > 5:
            dRH_dT = np.gradient(hall_coefficient, temperature)
            results['hall_coefficient_slope_300K'] = dRH_dT[np.argmin(np.abs(temperature - 300))]
        
        return results
    
    def analyze_thermopower(self, temperature: np.ndarray,
                          thermopower: np.ndarray) -> Dict[str, float]:
        """
        Analyze thermopower (Seebeck coefficient) data.
        
        Parameters:
        -----------
        temperature : np.ndarray
            Temperature values (K)
        thermopower : np.ndarray
            Thermopower values (μV/K)
            
        Returns: