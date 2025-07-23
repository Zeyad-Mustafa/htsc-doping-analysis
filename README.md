 Features 
 Core Analysis Capabilities

Doping Level Quantification: Calculate carrier concentrations from Hall effect measurements
Superconducting Dome Analysis: Determine optimal doping levels and dome boundaries
Phase Transition Detection: Identify structural and magnetic phase transitions
Transport Property Analysis: Comprehensive resistivity, Hall effect, and critical field analysis
 
 Supported Material Systems

Cuprates: YBCO, NCCO, BSCCO, and other copper-oxide superconductors
Iron Pnictides: LaFeAsO, BaFe₂As₂, and related compounds
Iron Chalcogenides: FeSe, FeTe, and their derivatives

🔬 Research Tools

Interactive phase diagram generation
Quantum critical point identification
Comparative analysis between hole- and electron-doped systems
Universal scaling relationship exploration
  
 Quick Start
Installation
bashgit clone https://github.com/Zeyad-Mustafa/htsc-doping-analysis.git
cd htsc-doping-analysis
pip install -r requirements.txt
Basic Usage
pythonfrom htsc_analysis import DopingAnalyzer, HTSCMaterial

# Create analyzer
analyzer = DopingAnalyzer()

# Define material
ybco = HTSCMaterial("YBCO", "cuprate", "hole")

# Add experimental data
ybco.add_data('tc_vs_doping', doping_levels, tc_values, 
              'Hole Doping', 'Tc (K)')

# Find optimal doping
optimal_doping, max_tc = analyzer.find_optimal_doping("YBCO")
print(f"Optimal doping: {optimal_doping:.3f}, Max Tc: {max_tc:.1f} K")
