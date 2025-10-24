# Route Analyzer Web GUI

A modern, interactive web interface for the Route Analyzer package, built with Streamlit.

## 🚀 Quick Start

### 1. Install GUI Dependencies
```bash
pip install -r requirements_gui.txt
```

### 2. Launch the GUI
```bash
# Option 1: Using the launcher script
python launch_gui.py

# Option 2: Direct Streamlit command
streamlit run launch_gui.py

# Option 3: From the project directory
streamlit run project/ra_gui.py
```

### 3. Open in Browser
The GUI will automatically open at `http://localhost:8501`

## 🎯 Features

### 📁 Data Upload
- **File Upload**: Drag & drop CSV files or specify folder paths
- **Column Mapping**: Configure X, Z, and Time column names
- **Parameter Tuning**: Adjust scale factor and motion threshold
- **Sample Data**: Load demo data for testing

### 🎯 Junction Editor
- **Interactive Plot**: Visual junction management with Plotly
- **Add/Remove**: Click to add junctions, delete with one click
- **Real-time Preview**: See trajectories and junctions together
- **Sample Junctions**: Load predefined junction configurations

### 📍 Zone Definition
- **Start Zones**: Define circular or rectangular start areas
- **End Zones**: Configure destination zones
- **Visual Feedback**: See zones overlaid on trajectory data
- **Multiple Zones**: Support for multiple start/end zones

### 📊 Analysis
- **Full Analysis**: Complete junction-based choice prediction
- **Quick Analysis**: Fast analysis with default parameters
- **Parameter Control**: Adjust all analysis settings
- **Real-time Status**: Live progress and completion indicators

### 📈 Visualization
- **Flow Graphs**: Interactive flow diagrams
- **Conditional Probabilities**: Heatmaps and probability matrices
- **Pattern Analysis**: Behavioral pattern identification
- **Start/End Analysis**: Completion rate and trajectory classification

### 💾 Export Results
- **Multiple Formats**: JSON, CSV, ZIP archives
- **Selective Export**: Choose which data to export
- **Download Ready**: Direct download from browser
- **Clipboard Support**: Copy results for sharing

## 🎨 Interface Overview

### Navigation Sidebar
- **Step-by-step workflow**: Data → Junctions → Zones → Analysis → Visualization → Export
- **Status indicators**: Visual feedback on completion status
- **Quick access**: Jump between any step

### Main Workspace
- **Responsive layout**: Adapts to different screen sizes
- **Interactive plots**: Zoom, pan, hover for details
- **Real-time updates**: Changes reflect immediately
- **Error handling**: Clear error messages and recovery options

## 🔧 Technical Details

### Architecture
- **Streamlit**: Modern web framework for Python
- **Plotly**: Interactive plotting and visualization
- **Modular Design**: Clean separation of concerns
- **Session State**: Persistent data across interactions

### Integration
- **Package Compatible**: Full Python package functionality preserved
- **Standalone Mode**: Can be used independently
- **Extensible**: Easy to add new features and visualizations

### Performance
- **Efficient Rendering**: Optimized for large datasets
- **Progressive Loading**: Load data incrementally
- **Caching**: Smart caching of analysis results

## 📱 Usage Examples

### Basic Workflow
1. **Upload Data**: Load your VR trajectory CSV files
2. **Define Junctions**: Add decision points interactively
3. **Set Zones**: Configure start and end areas
4. **Run Analysis**: Execute junction-based choice prediction
5. **Visualize**: Explore results with interactive plots
6. **Export**: Download results in your preferred format

### Advanced Usage
- **Custom Parameters**: Fine-tune analysis settings
- **Multiple Datasets**: Compare different experiments
- **Batch Processing**: Analyze multiple trajectory sets
- **Custom Visualizations**: Create specialized plots

## 🛠️ Development

### Adding New Features
1. **Extend RouteAnalyzerGUI class**: Add new methods
2. **Create new render methods**: Follow naming convention `render_*`
3. **Update navigation**: Add new steps to sidebar
4. **Test thoroughly**: Ensure compatibility with package mode

### Customization
- **Styling**: Modify CSS in the `render_header()` method
- **Layout**: Adjust column layouts and spacing
- **Colors**: Update Plotly color schemes
- **Icons**: Change emoji icons throughout

## 🔍 Troubleshooting

### Common Issues

**GUI won't start:**
```bash
# Check dependencies
pip install -r requirements_gui.txt

# Verify Python path
python -c "import streamlit; print('Streamlit OK')"
```

**Import errors:**
```bash
# Ensure you're in the right directory
cd /path/to/route_analyzer

# Check project structure
ls project/ra_gui.py
```

**Performance issues:**
- Reduce number of trajectories for testing
- Use sample data for initial setup
- Close other browser tabs

### Getting Help
- Check the console output for error messages
- Verify all dependencies are installed
- Ensure data files are in the correct format
- Try the sample data first

## 🎉 Benefits

### For Users
- **No coding required**: Visual interface for all operations
- **Interactive exploration**: Real-time parameter adjustment
- **Professional results**: Publication-ready visualizations
- **Easy sharing**: Export and share results easily

### For Developers
- **Package integration**: Use as Python package or GUI
- **Extensible**: Add new analysis methods easily
- **Modern stack**: Built with current web technologies
- **Cross-platform**: Works on Windows, Mac, Linux

## 🔮 Future Enhancements

- **Real-time analysis**: Live updates as data changes
- **Collaborative features**: Share sessions between users
- **Advanced visualizations**: 3D plots, animations
- **Machine learning**: Automated pattern detection
- **Cloud deployment**: Host on cloud platforms
