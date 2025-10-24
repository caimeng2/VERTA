# Route Analyzer GUI

A web-based graphical interface for the Route Analyzer package, built with Streamlit.

## Features

### 🖥️ Interactive Interface
- **Data Loading**: Upload CSV files or specify folder paths
- **Junction Management**: Add, remove, and edit junctions interactively
- **Zone Definition**: Define start and end zones visually
- **Real-time Analysis**: Run analysis with live parameter adjustment
- **Export Results**: Download results in multiple formats

### 🎯 Key Capabilities
- **Interactive Junction Editor**: Click to add junctions, drag to reposition
- **Visual Zone Definition**: Draw start/end zones with different shapes
- **Live Analysis Preview**: See results as you adjust parameters
- **Multiple Export Formats**: JSON, CSV, ZIP archives
- **Real-time Statistics**: Live updates of analysis metrics

## Installation

### Option 1: Full Installation (Recommended)
```bash
# Install everything including GUI
python setup.py
```

### Option 2: Manual Installation
```bash
# Install core package
pip install -e .

# Install GUI dependencies
pip install -r requirements_gui.txt
```

## Usage

### GUI Mode
```bash
# Method 1: Using the main package
python -m project.route_analyzer --gui

# Method 2: Using the launcher
python ra_gui_launcher.py

# Method 3: Direct streamlit
streamlit run project/ra_gui.py
```

### CLI Mode (Still Available)
```bash
# All existing CLI commands still work
python -m project.route_analyzer discover --input data/ --out results/
python -m project.route_analyzer predict --input data/ --junctions 500 300 30
```

## GUI Workflow

### 1. Data Loading
- Upload CSV files or specify a folder path
- Configure column mapping (X, Z, Time columns)
- Set scale factor and motion threshold
- Load trajectories with real-time feedback

### 2. Junction Management
- **Add Junctions**: Click on the map or use coordinates
- **Edit Junctions**: Modify position and radius
- **Remove Junctions**: Select and delete unwanted junctions
- **Import/Export**: Save and load junction configurations

### 3. Zone Definition
- **Start Zones**: Define where trajectories begin
  - Circle: Center point + radius
  - Rectangle: Min/Max coordinates
- **End Zones**: Define completion areas
  - Rectangle: Target area coordinates
  - Circle: Center point + radius

### 4. Analysis & Visualization
- **Parameter Adjustment**: Real-time parameter tuning
- **Analysis Execution**: Run complete analysis pipeline
- **Results Display**: Interactive charts and tables
- **Pattern Analysis**: Behavioral pattern identification

### 5. Export Results
- **JSON Export**: Complete analysis results
- **CSV Export**: Summary tables
- **ZIP Archive**: All files and visualizations
- **Visualization Export**: Selected charts and plots

## GUI Pages

### 📁 Data Loading & Configuration
- File upload interface
- Column mapping configuration
- Data validation and preview
- Loading status and statistics

### 🔗 Junction Management
- Interactive junction editor
- Real-time trajectory visualization
- Junction property editing
- Import/export functionality

### 🎯 Zone Definition
- Visual zone drawing tools
- Start/end zone configuration
- Zone validation and preview
- Completion rate estimation

### 📊 Analysis & Visualization
- Parameter adjustment interface
- Real-time analysis execution
- Interactive result visualization
- Pattern analysis and statistics

### 📤 Export Results
- Multiple export formats
- Customizable export options
- Batch export capabilities
- Download management

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python package integration
- **Visualization**: Plotly interactive charts
- **Data Handling**: Pandas/NumPy integration

### Dependencies
- `streamlit`: Web framework
- `plotly`: Interactive visualizations
- `pandas`: Data manipulation
- `numpy`: Numerical computing

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

### Common Issues

**GUI won't start:**
```bash
# Check dependencies
pip install -r requirements_gui.txt

# Try launcher
python ra_gui_launcher.py
```

**Import errors:**
```bash
# Reinstall package
pip install -e .
```

**Browser issues:**
- Clear browser cache
- Try different browser
- Check firewall settings

### Performance Tips
- Limit trajectory display to 50 for better performance
- Use smaller datasets for initial testing
- Close unused browser tabs
- Ensure sufficient RAM (4GB+ recommended)

## Integration with Package

The GUI maintains full compatibility with the Python package:

```python
# Import and use programmatically
from project.ra_gui import RouteAnalyzerGUI
from project.ra_prediction import analyze_junction_choice_patterns

# Use GUI class
gui = RouteAnalyzerGUI()

# Or use core functions directly
results = analyze_junction_choice_patterns(trajectories, chain_df, junctions)
```

## Development

### Adding New Features
1. Modify `project/ra_gui.py`
2. Update GUI pages and methods
3. Test with different datasets
4. Update documentation

### Customization
- Modify page layouts in `ra_gui.py`
- Add new visualization types
- Extend export formats
- Customize UI themes

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the main package documentation
3. Check browser console for errors
4. Verify all dependencies are installed

---

**Note**: The GUI is designed to work alongside the existing CLI interface. All core functionality remains available through both interfaces.
