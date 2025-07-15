"""
High Temperature Superconductor Doping Analysis
Core classes for analyzing electron and hole doped HTSC materials
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import warnings

class HTSCMaterial:
    """Base class for high temperature superconductor materials"""
    
    def __init__(self, name: str, material_type: str, doping_type: str):
        self.name = name
        self.material_type = material_type  # 'cuprate', 'iron_pnictide', 'iron_chalcogenide'
        self.doping_type = doping_type  # 'hole', 'electron'
        self.data = {}
        self.properties = {}
        
    def add_data(self, data_type: str, x_data: np.ndarray, y_data: np.ndarray, 
                 x_label: str = "", y_label: str = ""):
        """Add experimental data to the material"""
        self.data[data_type] = {
            'x': x_data,
            'y': y_data,
            'x_label': x_label,
            'y_label': y_label
        }
    
    def get_data(self, data_type: str) -> Dict:
        """Retrieve data for a specific measurement type"""
        return self.data.get(data_type, {})

class DopingAnalyzer:
    """Main class for analyzing doping effects in HTSC materials"""
    
    def __init__(self):
        self.materials = {}
        self.phase_diagrams = {}
        
    def add_material(self, material: HTSCMaterial):
        """Add a material to the analyzer"""
        self.materials[material.name] = material
        
    def calculate_carrier_concentration(self, material_name: str, 
                                      hall_coefficient: np.ndarray) -> np.ndarray:
        """Calculate carrier concentration from Hall coefficient"""
        e = 1.602e-19  # elementary charge
        return 1 / (e * hall_coefficient)
    
    def estimate_doping_level(self, material_name: str, 
                            carrier_concentration: np.ndarray) -> np.ndarray:
        """Estimate doping level from carrier concentration"""
        material = self.materials.get(material_name)
        if not material:
            raise ValueError(f"Material {material_name} not found")
            
        # Reference carrier concentrations (material-specific)
        ref_concentrations = {
            'cuprate': 1e28,  # m^-3
            'iron_pnictide': 1e28,
            'iron_chalcogenide': 1e28
        }
        
        ref_conc = ref_concentrations.get(material.material_type, 1e28)
        return carrier_concentration / ref_conc
    
    def find_optimal_doping(self, material_name: str) -> Tuple[float, float]:
        """Find optimal doping level where Tc is maximum"""
        material = self.materials.get(material_name)
        if not material or 'tc_vs_doping' not in material.data:
            raise ValueError(f"Tc vs doping data not available for {material_name}")
            
        data = material.data['tc_vs_doping']
        max_idx = np.argmax(data['y'])
        optimal_doping = data['x'][max_idx]
        max_tc = data['y'][max_idx]
        
        return optimal_doping, max_tc
    
    def calculate_superconducting_dome(self, material_name: str) -> Dict:
        """Calculate superconducting dome parameters"""
        material = self.materials.get(material_name)
        if not material or 'tc_vs_doping' not in material.data:
            raise ValueError(f"Tc vs doping data not available for {material_name}")
            
        data = material.data['tc_vs_doping']
        doping = data['x']
        tc = data['y']
        
        # Find dome boundaries (where Tc drops to near zero)
        threshold = 0.05 * np.max(tc)
        superconducting_region = tc > threshold
        
        if not np.any(superconducting_region):
            return {'underdoped_limit': None, 'overdoped_limit': None, 'width': None}
            
        sc_indices = np.where(superconducting_region)[0]
        underdoped_limit = doping[sc_indices[0]]
        overdoped_limit = doping[sc_indices[-1]]
        dome_width = overdoped_limit - underdoped_limit
        
        return {
            'underdoped_limit': underdoped_limit,
            'overdoped_limit': overdoped_limit,
            'width': dome_width
        }

class PhaseTransitionAnalyzer:
    """Analyze phase transitions in doped HTSC materials"""
    
    def __init__(self):
        self.transition_temperatures = {}
        
    def analyze_structural_transition(self, temperature: np.ndarray, 
                                    lattice_parameter: np.ndarray) -> Dict:
        """Analyze structural phase transitions"""
        # Find derivative to locate transition
        derivative = np.gradient(lattice_parameter, temperature)
        
        # Find peaks in derivative (transition points)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(np.abs(derivative), height=np.std(derivative))
        
        transition_temps = temperature[peaks] if len(peaks) > 0 else []
        
        return {
            'transition_temperatures': transition_temps,
            'derivative': derivative,
            'peaks': peaks
        }
    
    def analyze_magnetic_transition(self, temperature: np.ndarray, 
                                  susceptibility: np.ndarray) -> Dict:
        """Analyze magnetic phase transitions"""
        # Curie-Weiss fit for high temperature region
        def curie_weiss(T, C, theta):
            return C / (T - theta)
        
        # Use high temperature data for fitting
        high_temp_mask = temperature > 0.7 * np.max(temperature)
        
        try:
            popt, _ = curve_fit(curie_weiss, 
                              temperature[high_temp_mask], 
                              susceptibility[high_temp_mask])
            curie_constant, weiss_temperature = popt
        except:
            curie_constant, weiss_temperature = None, None
        
        return {
            'curie_constant': curie_constant,
            'weiss_temperature': weiss_temperature,
            'fit_function': curie_weiss
        }

class TransportAnalyzer:
    """Analyze transport properties in doped HTSC materials"""
    
    def __init__(self):
        self.transport_data = {}
        
    def analyze_resistivity(self, temperature: np.ndarray, 
                          resistivity: np.ndarray, 
                          doping_level: float) -> Dict:
        """Analyze resistivity vs temperature behavior"""
        
        # Find superconducting transition
        tc_onset = self._find_tc_onset(temperature, resistivity)
        tc_zero = self._find_tc_zero(temperature, resistivity)
        
        # Analyze normal state behavior
        normal_state_mask = temperature > tc_onset if tc_onset else temperature > 0
        
        # Linear fit for normal state resistivity
        if np.any(normal_state_mask):
            normal_temp = temperature[normal_state_mask]
            normal_rho = resistivity[normal_state_mask]
            
            # Linear fit: rho = rho_0 + A*T
            coeffs = np.polyfit(normal_temp, normal_rho, 1)
            residual_resistivity = coeffs[1]  # rho_0
            temp_coefficient = coeffs[0]      # A
        else:
            residual_resistivity = None
            temp_coefficient = None
        
        return {
            'tc_onset': tc_onset,
            'tc_zero': tc_zero,
            'residual_resistivity': residual_resistivity,
            'temperature_coefficient': temp_coefficient,
            'doping_level': doping_level
        }
    
    def _find_tc_onset(self, temperature: np.ndarray, 
                      resistivity: np.ndarray) -> Optional[float]:
        """Find superconducting transition onset temperature"""
        # Find where resistivity starts dropping rapidly
        derivative = np.gradient(resistivity, temperature)
        
        # Look for steepest negative slope
        min_derivative_idx = np.argmin(derivative)
        
        if derivative[min_derivative_idx] < -0.1 * np.max(resistivity) / np.max(temperature):
            return temperature[min_derivative_idx]
        return None
    
    def _find_tc_zero(self, temperature: np.ndarray, 
                     resistivity: np.ndarray) -> Optional[float]:
        """Find zero resistance temperature"""
        # Find where resistivity becomes negligible
        threshold = 0.01 * np.max(resistivity)
        zero_mask = resistivity < threshold
        
        if np.any(zero_mask):
            return temperature[np.where(zero_mask)[0][-1]]
        return None
    
    def calculate_coherence_length(self, tc: float, 
                                 upper_critical_field: float) -> float:
        """Calculate coherence length from Tc and Hc2"""
        phi_0 = 2.067e-15  # flux quantum (Wb)
        return np.sqrt(phi_0 / (2 * np.pi * upper_critical_field))

class ComparativeAnalyzer:
    """Compare different doping types and materials"""
    
    def __init__(self):
        self.comparison_data = {}
        
    def compare_doping_types(self, hole_doped_material: HTSCMaterial, 
                           electron_doped_material: HTSCMaterial) -> Dict:
        """Compare hole and electron doped materials"""
        
        comparison = {
            'hole_doped': {
                'name': hole_doped_material.name,
                'type': hole_doped_material.doping_type,
                'material_type': hole_doped_material.material_type
            },
            'electron_doped': {
                'name': electron_doped_material.name,
                'type': electron_doped_material.doping_type,
                'material_type': electron_doped_material.material_type
            }
        }
        
        # Compare Tc values if available
        for material_key, material in [('hole_doped', hole_doped_material), 
                                     ('electron_doped', electron_doped_material)]:
            if 'tc_vs_doping' in material.data:
                data = material.data['tc_vs_doping']
                max_tc = np.max(data['y'])
                optimal_doping = data['x'][np.argmax(data['y'])]
                
                comparison[material_key]['max_tc'] = max_tc
                comparison[material_key]['optimal_doping'] = optimal_doping
        
        return comparison
    
    def analyze_universal_trends(self, materials: List[HTSCMaterial]) -> Dict:
        """Analyze universal trends across different materials"""
        
        trends = {
            'hole_doped': [],
            'electron_doped': []
        }
        
        for material in materials:
            if 'tc_vs_doping' in material.data:
                data = material.data['tc_vs_doping']
                max_tc = np.max(data['y'])
                optimal_doping = data['x'][np.argmax(data['y'])]
                
                material_info = {
                    'name': material.name,
                    'material_type': material.material_type,
                    'max_tc': max_tc,
                    'optimal_doping': optimal_doping
                }
                
                trends[material.doping_type].append(material_info)
        
        return trends

# Example usage and data generation functions
def generate_sample_data():
    """Generate sample data for testing"""
    
    # Sample doping levels
    hole_doping = np.linspace(0, 0.3, 30)
    electron_doping = np.linspace(0, 0.2, 25)
    
    # Sample Tc vs doping for hole-doped cuprate (YBCO-like)
    tc_hole = 90 * np.exp(-((hole_doping - 0.16)**2) / (2 * 0.05**2))
    tc_hole[hole_doping < 0.05] = 0
    tc_hole[hole_doping > 0.25] = 0
    
    # Sample Tc vs doping for electron-doped cuprate (NCCO-like)
    tc_electron = 25 * np.exp(-((electron_doping - 0.15)**2) / (2 * 0.03**2))
    tc_electron[electron_doping < 0.12] = 0
    tc_electron[electron_doping > 0.18] = 0
    
    return {
        'hole_doping': hole_doping,
        'tc_hole': tc_hole,
        'electron_doping': electron_doping,
        'tc_electron': tc_electron
    }

if __name__ == "__main__":
    # Example usage
    analyzer = DopingAnalyzer()
    
    # Create sample materials
    ybco = HTSCMaterial("YBCO", "cuprate", "hole")
    ncco = HTSCMaterial("NCCO", "cuprate", "electron")
    
    # Add sample data
    sample_data = generate_sample_data()
    
    ybco.add_data('tc_vs_doping', 
                  sample_data['hole_doping'], 
                  sample_data['tc_hole'],
                  'Hole Doping', 'Tc (K)')
    
    ncco.add_data('tc_vs_doping', 
                  sample_data['electron_doping'], 
                  sample_data['tc_electron'],
                  'Electron Doping', 'Tc (K)')
    
    # Add materials to analyzer
    analyzer.add_material(ybco)
    analyzer.add_material(ncco)
    
    # Analyze optimal doping
    opt_doping_ybco, max_tc_ybco = analyzer.find_optimal_doping("YBCO")
    opt_doping_ncco, max_tc_ncco = analyzer.find_optimal_doping("NCCO")
    
    print(f"YBCO - Optimal doping: {opt_doping_ybco:.3f}, Max Tc: {max_tc_ybco:.1f} K")
    print(f"NCCO - Optimal doping: {opt_doping_ncco:.3f}, Max Tc: {max_tc_ncco:.1f} K")
    
    # Analyze superconducting dome
    dome_ybco = analyzer.calculate_superconducting_dome("YBCO")
    dome_ncco = analyzer.calculate_superconducting_dome("NCCO")
    
    print(f"\nYBCO dome width: {dome_ybco['width']:.3f}")
    print(f"NCCO dome width: {dome_ncco['width']:.3f}")