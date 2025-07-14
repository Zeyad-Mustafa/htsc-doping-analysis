"""
Phase Diagrams Module for High-Temperature Superconductors

This module provides tools for generating and analyzing phase diagrams
of HTSC materials, including temperature-doping phase diagrams,
superconducting domes, and quantum critical point analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PhaseDiagram:
    """
    Class for generating and analyzing phase diagrams of HTSC materials.
    """
    
    def __init__(self, material_system: str):
        """
        Initialize PhaseDiagram object.
        
        Parameters:
        -----------
        material_system : str
            Name of the material system (e.g., 'YBCO', 'LSCO', 'BaFe2As2')
        """
        self.material_system = material_system
        self.phase_data = {}
        
        # Color scheme for different phases
        self.phase_colors = {
            'antiferromagnetic': '#FF6B6B',
            'superconducting': '#4ECDC4',
            'metallic': '#45B7D1',
            'pseudogap': '#FFA07A',
            'coexistence': '#98D8C8'
        }
    
    def generate_tx_diagram(self, data: pd.DataFrame, 
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Generate Temperature vs Doping phase diagram.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing doping levels and transition temperatures
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            Generated phase diagram
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        doping = data['doping_level'].values
        
        # Plot superconducting transition
        if 'tc_k' in data.columns:
            tc_data = data['tc_k'].values
            sc_mask = tc_data > 0
            ax.fill_between(doping[sc_mask], 0, tc_data[sc_mask], 
                          alpha=0.7, color=self.phase_colors['superconducting'],
                          label='Superconducting')
        
        # Plot antiferromagnetic transition
        if 'af_transition_temp_k' in data.columns:
            af_data = data['af_transition_temp_k'].values
            af_mask = af_data > 0
            ax.plot(doping[af_mask], af_data[af_mask], 'o-', 
                   color=self.phase_colors['antiferromagnetic'],
                   linewidth=2, markersize=6, label='Antiferromagnetic')
        
        # Plot pseudogap crossover
        if 'pseudogap_temp_k' in data.columns:
            pg_data = data['pseudogap_temp_k'].values
            pg_mask = pg_data > 0
            ax.plot(doping[pg_mask], pg_data[pg_mask], 's--', 
                   color=self.phase_colors['pseudogap'],
                   linewidth=2, markersize=5, label='Pseudogap')
        
        # Plot structural transition (for iron-based materials)
        if 'structural_transition_k' in data.columns:
            struct_data = data['structural_transition_k'].values
            struct_mask = struct_data > 0
            ax.plot(doping[struct_mask], struct_data[struct_mask], '^-', 
                   color='purple', linewidth=2, markersize=6, 
                   label='Structural')
        
        ax.set_xlabel('Doping Level', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title(f'{self.material_system} Phase Diagram', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_superconducting_dome(self, tc_data: np.ndarray, 
                                 doping_levels: np.ndarray,
                                 fit_curve: bool = True,
                                 figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot superconducting dome with optional curve fitting.
        
        Parameters:
        -----------
        tc_data : np.ndarray
            Critical temperature values
        doping_levels : np.ndarray
            Doping level values
        fit_curve : bool
            Whether to fit a curve to the data
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Superconducting dome plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot experimental data
        ax.plot(doping_levels, tc_data, 'o', markersize=8, 
               color=self.phase_colors['superconducting'], 
               label='Experimental Data')
        
        # Fit curve if requested
        if fit_curve and len(tc_data) > 4:
            # Parabolic fit for dome shape
            def dome_function(x, a, x0, tc_max):
                return np.maximum(0, a * (x - x0)**2 + tc_max)
            
            try:
                # Initial guess
                max_idx = np.argmax(tc_data)
                x0_guess = doping_levels[max_idx]
                tc_max_guess = tc_data[max_idx]
                a_guess = -tc_max_guess / (0.1)**2  # Rough estimate
                
                popt, _ = curve_fit(dome_function, doping_levels, tc_data,
                                  p0=[a_guess, x0_guess, tc_max_guess])
                
                # Generate smooth curve
                x_smooth = np.linspace(doping_levels.min(), doping_levels.max(), 200)
                y_smooth = dome_function(x_smooth, *popt)
                
                ax.plot(x_smooth, y_smooth, '-', linewidth=2, 
                       color='red', label='Fitted Curve')
                
                # Add fit parameters to plot
                ax.text(0.05, 0.95, f'Optimal doping: {popt[1]:.3f}\nMax Tc: {popt[2]:.1f} K',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Curve fitting failed: {e}")
        
        ax.set_xlabel('Doping Level', fontsize=12)
        ax.set_ylabel('Critical Temperature (K)', fontsize=12)
        ax.set_title(f'{self.material_system} Superconducting Dome', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def identify_quantum_critical_point(self, data: pd.DataFrame,
                                      temperature_range: Tuple[float, float] = (0, 50)) -> Dict[str, float]:
        """
        Identify quantum critical points in the phase diagram.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Phase diagram data
        temperature_range : tuple
            Temperature range to search for QCP
            
        Returns:
        --------
        Dict[str, float]
            Quantum critical point information
        """
        qcp_info = {}
        
        # Look for end of antiferromagnetic phase
        if 'af_transition_temp_k' in data.columns:
            af_data = data[data['af_transition_temp_k'] > 0]
            if len(af_data) > 0:
                # Find where AF temperature approaches zero
                af_interp = interp1d(af_data['doping_level'], af_data['af_transition_temp_k'],
                                   kind='linear', fill_value='extrapolate')
                
                # Find doping level where AF temperature would be zero
                doping_range = np.linspace(af_data['doping_level'].min(),
                                         af_data['doping_level'].max() + 0.1, 100)
                af_temps = af_interp(doping_range)
                
                zero_crossing_idx = np.where(af_temps <= 0)[0]
                if len(zero_crossing_idx) > 0:
                    qcp_doping = doping_range[zero_crossing_idx[0]]
                    qcp_info['antiferromagnetic_qcp'] = qcp_doping
        
        # Look for superconducting quantum critical point
        if 'tc_k' in data.columns:
            sc_data = data[data['tc_k'] > 0]
            if len(sc_data) > 0:
                # End of superconducting dome
                max_doping_sc = sc_data['doping_level'].max()
                qcp_info['superconducting_qcp_upper'] = max_doping_sc
                
                min_doping_sc = sc_data['doping_level'].min()
                qcp_info['superconducting_qcp_lower'] = min_doping_sc
        
        return qcp_info
    
    def analyze_dome_asymmetry(self, tc_data: np.ndarray, 
                             doping_levels: np.ndarray) -> Dict[str, float]:
        """
        Analyze asymmetry of the superconducting dome.
        
        Parameters:
        -----------
        tc_data : np.ndarray
            Critical temperature values
        doping_levels : np.ndarray
            Doping level values
            
        Returns:
        --------
        Dict[str, float]
            Asymmetry analysis results
        """
        # Find optimal doping (maximum Tc)
        max_idx = np.argmax(tc_data)
        optimal_doping = doping_levels[max_idx]
        max_tc = tc_data[max_idx]
        
        # Find dome boundaries (where Tc drops to 10% of maximum)
        threshold = 0.1 * max_tc
        sc_indices = np.where(tc_data > threshold)[0]
        
        if len(sc_indices) < 2:
            return {'asymmetry': 0, 'left_width': 0, 'right_width': 0}
        
        left_boundary = doping_levels[sc_indices[0]]
        right_boundary = doping_levels[sc_indices[-1]]
        
        left_width = optimal_doping - left_boundary
        right_width = right_boundary - optimal_doping
        
        asymmetry = (right_width - left_width) / (right_width + left_width)
        
        return {
            'asymmetry': asymmetry,
            'left_width': left_width,
            'right_width': right_width,
            'optimal_doping': optimal_doping,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary
        }
    
    def create_3d_phase_diagram(self, data: pd.DataFrame,
                               temperature_col: str = 'temperature',
                               doping_col: str = 'doping_level',
                               property_col: str = 'tc_k') -> plt.Figure:
        """
        Create 3D phase diagram visualization.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for 3D plotting
        temperature_col : str
            Column name for temperature
        doping_col : str
            Column name for doping
        property_col : str
            Column name for the property to plot
            
        Returns:
        --------
        plt.Figure
            3D phase diagram
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for surface plot
        T = data[temperature_col].values
        x = data[doping_col].values
        z = data[property_col].values
        
        # Create regular grid
        Ti = np.linspace(T.min(), T.max(), 50)
        xi = np.linspace(x.min(), x.max(), 50)
        Ti, xi = np.meshgrid(Ti, xi)
        
        # Interpolate data onto regular grid
        zi = griddata((T, x), z, (Ti, xi), method='linear')
        
        # Plot surface
        surf = ax.plot_surface(Ti, xi, zi, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Doping Level')
        ax.set_zlabel(property_col.replace('_', ' ').title())
        ax.set_title(f'{self.material_system} 3D Phase Diagram')
        
        return fig
    
    def plot_comparative_domes(self, datasets: List[Dict],
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot multiple superconducting domes for comparison.
        
        Parameters:
        -----------
        datasets : List[Dict]
            List of datasets, each containing 'doping', 'tc', and 'label'
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Comparative dome plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        for i, dataset in enumerate(datasets):
            doping = dataset['doping']
            tc = dataset['tc']
            label = dataset['label']
            
            ax.plot(doping, tc, 'o-', color=colors[i], linewidth=2,
                   markersize=6, label=label)
        
        ax.set_xlabel('Doping Level', fontsize=12)
        ax.set_ylabel('Critical Temperature (K)', fontsize=12)
        ax.set_title('Comparative Superconducting Domes', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def calculate_phase_volume(self, data: pd.DataFrame,
                              phase_column: str = 'phase') -> Dict[str, float]:
        """
        Calculate the volume of different phases in the phase diagram.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Phase diagram data
        phase_column : str
            Column containing phase information
            
        Returns:
        --------
        Dict[str, float]
            Phase volumes (as fractions)
        """
        if phase_column not in data.columns:
            return {}
        
        phase_counts = data[phase_column].value_counts()
        total_points = len(data)
        
        phase_volumes = {}
        for phase, count in phase_counts.items():
            phase_volumes[phase] = count / total_points
        
        return phase_volumes
    
    def plot_phase_evolution(self, data: pd.DataFrame,
                           property_name: str = 'tc_k',
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot evolution of a property across the phase diagram.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Phase diagram data
        property_name : str
            Name of property to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Phase evolution plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        doping = data['doping_level'].values
        prop_values = data[property_name].values
        
        # Top plot: Property vs doping
        ax1.plot(doping, prop_values, 'o-', linewidth=2, markersize=6)
        ax1.set_ylabel(property_name.replace('_', ' ').title())
        ax1.set_title(f'{self.material_system}: {property_name} Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Derivative to show phase transitions
        if len(prop_values) > 3:
            derivative = np.gradient(prop_values, doping)
            ax2.plot(doping, derivative, 'r-', linewidth=2, label='Derivative')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_ylabel(f'd({property_name})/d(doping)')
            ax2.set_xlabel('Doping Level')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return fig