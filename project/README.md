# Route Analyzer

A comprehensive toolkit to analyze initial route choices from x–z movement trajectories: discover branch directions, assign trajectories to branches, compute timing metrics, visualize results, and predict future route choices based on behavioral patterns. Includes both command-line interface (CLI) and interactive web GUI.

## Installation

```bash
# Core installation
pip install .

# With optional extras
pip install .[yaml]      # YAML configuration support
pip install .[parquet]    # Parquet file format support  
pip install .[gui]        # GUI dependencies (streamlit, plotly)
```

This installs a console script `route-analyzer` and enables the web GUI.

## 🖥️ Web GUI

For an interactive, user-friendly interface, try the web-based GUI:

```bash
# Install GUI dependencies
pip install -r requirements_gui.txt

# Launch the web interface
python launch_gui.py
# or
streamlit run launch_gui.py
```

The GUI provides:
- **Interactive junction editor** with drag-and-drop functionality
- **Visual zone definition** for start/end points
- **Real-time analysis** with live parameter adjustment
- **Interactive visualizations** with Plotly charts
- **Gaze and physiological analysis** (head yaw, pupil dilation)
- **Flow graph generation** and conditional probability analysis
- **Pattern recognition** and behavioral insights
- **Export capabilities** in multiple formats (JSON, CSV, ZIP)
- **Multi-junction analysis** with evacuation planning features

See [GUI_README.md](GUI_README.md) for detailed GUI documentation.

## CLI Commands

The Route Analyzer provides 6 main commands for different types of analysis:

### 1. Discover Branches

Discover branch directions from trajectory data using clustering algorithms:

```bash
route-analyzer discover \
  --input ./data \
  --glob "*.csv" \
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \
  --scale 0.2 \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --decision_mode hybrid \
  --cluster_method auto \
  --out ./outputs
```

**Key Parameters:**
- `--cluster_method`: Choose from `kmeans`, `auto`, or `dbscan`
- `--decision_mode`: `pathlen`, `radial`, or `hybrid`
- `--k`: Number of clusters (for kmeans)
- `--r_outer`: Outer radius for radial decision mode
- `--linger_delta`: Distance beyond junction for decision detection

### 2. Assign Branches

Assign new trajectories to previously discovered branch centers:

```bash
route-analyzer assign \
  --input ./new_data \
  --columns x=X,z=Z,t=time \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --centers ./outputs/branch_centers.npy \
  --out ./outputs/new_assignments
```

### 3. Compute Metrics

Calculate timing and speed metrics for trajectories:

```bash
route-analyzer metrics \
  --input ./data \
  --columns x=X,z=Z,t=time \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --decision_mode radial --r_outer 30 \
  --out ./outputs
```

**Metrics Computed:**
- Time to travel specified path length after junction
- Speed through junction (entry, exit, average transit)
- Time between regions (if `--regions` specified)

### 4. Gaze Analysis

Analyze head movement and physiological data at decision points:

```bash
route-analyzer gaze \
  --input ./gaze_data \
  --columns x=X,z=Z,t=time,yaw=HeadYaw,pupil=PupilDilation \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --physio_window 3.0 \
  --out ./gaze_outputs
```

**Analysis Features:**
- Head yaw direction at decision points
- Pupil dilation trajectory analysis
- Physiological metrics (heart rate, etc.) at junctions
- Gaze-movement consistency analysis

### 5. Predict Choices

Analyze behavioral patterns and predict future route choices:

```bash
route-analyzer predict \
  --input ./data \
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \
  --scale 0.2 \
  --junctions 520 330 20  600 400 20  700 450 20 \
  --r_outer_list 30 32 30 \
  --distance 100 \
  --decision_mode hybrid \
  --cluster_method auto \
  --analyze_sequences \
  --predict_examples 50 \
  --out ./outputs/prediction
```

**Prediction Features:**
- Behavioral pattern recognition (preferred, learned, direct)
- Conditional probability analysis between junctions
- Route sequence analysis
- Confidence scoring for predictions
- Concrete prediction examples

### 6. Enhanced Chain Analysis

Multi-junction analysis with evacuation planning features:

```bash
route-analyzer chain-enhanced \
  --input ./data \
  --columns x=X,z=Z,t=time \
  --junctions 520 330 20  600 400 20  700 450 20 \
  --r_outer_list 30 32 30 \
  --distance 100 \
  --evacuation_analysis \
  --generate_recommendations \
  --risk_assessment \
  --out ./outputs/chain_analysis
```

**Enhanced Features:**
- Flow graph generation
- Evacuation efficiency analysis
- Risk assessment metrics
- Traffic flow recommendations

## Output Files Reference

Each command generates specific output files:

### Discover Command
- `branch_assignments.csv` - Main branch assignments
- `branch_assignments_all.csv` - All assignments including outliers
- `branch_centers.npy` / `branch_centers.json` - Branch center coordinates
- `branch_summary.csv` - Branch count statistics with entropy
- `Branch_Directions.png` - Visual plot of branch directions
- `Branch_Counts.png` - Bar chart of branch frequencies
- `Decision_Intercepts.png` - Trajectory decision points visualization
- `Decision_Map.png` - Overview map of decisions

### Assign Command
- `branch_assignments.csv` - New trajectory assignments
- `run_args.json` - Command parameters used

### Metrics Command
- `timing_and_speed_metrics.csv` - Comprehensive timing and speed data
- `run_args.json` - Command parameters used

### Gaze Command
- `gaze_analysis.csv` - Head yaw analysis at decision points
- `physiological_analysis.csv` - Physiological metrics at junctions
- `pupil_trajectory_analysis.csv` - Pupil dilation trajectory data
- `gaze_consistency_report.json` - Gaze-movement alignment statistics
- `Gaze_Directions.png` - Head movement visualization
- `Physiological_Analysis.png` - Physiological metrics by branch
- `Pupil_Trajectory_Analysis.png` - Pupil dilation plots

### Predict Command
- `choice_pattern_analysis.json` - Complete pattern analysis results
- `choice_patterns.png` - Behavioral pattern visualization
- `transition_heatmap.png` - Junction transition probabilities
- `prediction_examples.json` - Concrete prediction examples
- `prediction_confidence.png` - Confidence analysis plots
- `sequence_analysis.json` - Route sequence analysis (if `--analyze_sequences`)
- `analysis_summary.json` - High-level summary with recommendations

### Chain-Enhanced Command
- `Chain_Overview.png` - Multi-junction trajectory overview
- `Chain_SmallMultiples.png` - Detailed junction-by-junction view
- `Flow_Graph_Map.png` - Flow diagram between junctions
- `Per_Junction_Flow_Graph.png` - Individual junction flow analysis
- `branch_decisions_chain.csv` - Complete decision chain data

## Behavioral Pattern Analysis

The Route Analyzer identifies three types of behavioral patterns:

### Pattern Types

- **Preferred Patterns** (probability ≥ 0.7): Strong behavioral preferences that are highly predictable
- **Learned Patterns** (probability 0.5-0.7): Patterns that develop over time as participants learn the environment  
- **Direct Patterns** (probability 0.3-0.5): Basic route choices without strong preferences

### Example Analysis Results

```json
{
  "summary": {
    "total_sequences": 150,
    "total_transitions": 300,
    "unique_patterns": 12,
    "junctions_analyzed": 3
  },
  "pattern_types": {
    "preferred": 3,
    "learned": 5,
    "direct": 4
  },
  "top_patterns": [
    {
      "from_junction": 0,
      "to_junction": 1,
      "from_branch": 1,
      "to_branch": 2,
      "probability": 0.85,
      "confidence": 0.92,
      "sample_size": 23,
      "pattern_type": "preferred"
    }
  ]
}
```

### Applications

The analysis can help identify:
- Which junctions are most predictable vs. variable
- How participants adapt to the VR environment over time
- Optimal junction designs based on user behavior
- Potential traffic bottlenecks or flow issues
- Evacuation route efficiency and safety

## Configuration

Pass `--config path/to/config.yaml` to any subcommand. Keys under `defaults:` apply to all subcommands; subcommand-specific blocks (`discover:`, `assign:`, `metrics:`) override defaults.

Example `config.yaml`:
```yaml
defaults:
  glob: "*.csv"
  columns: { x: "Headset.Head.Position.X", z: "Headset.Head.Position.Z", t: "Time" }
  scale: 0.2
  motion_threshold: 0.001
  radius: 20
  distance: 100
  epsilon: 0.015
  junction: [520, 330]

discover:
  decision_mode: hybrid
  r_outer: 30
  linger_delta: 2.0
  cluster_method: dbscan
  angle_eps: 15
  show_paths: true
  plot_intercepts: true
```

## Dependencies

### Core Dependencies
- **numpy** (≥1.20.0) - Numerical computations
- **pandas** (≥1.3.0) - Data manipulation and analysis
- **matplotlib** (≥3.3.0) - Static plotting and visualization
- **scikit-learn** (≥1.0.0) - Machine learning algorithms
- **tqdm** (≥4.60.0) - Progress bars
- **seaborn** (≥0.12.0) - Statistical data visualization

### Optional Dependencies

Install with `pip install .[extras]`:

- **yaml** - YAML configuration file support (`pyyaml`)
- **parquet** - Parquet file format support (`pyarrow`)
- **gui** - Web GUI dependencies (`streamlit`, `plotly`)

### GUI-Specific Dependencies

For the web interface, install:
```bash
pip install -r requirements_gui.txt
```

Additional GUI packages:
- **streamlit** (≥1.28.0) - Web framework
- **plotly** (≥5.15.0) - Interactive plotting
- **streamlit-option-menu** (≥0.3.0) - Enhanced UI components
- **streamlit-aggrid** (≥0.3.0) - Data grid components

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the project directory
cd /path/to/route_analyzer/project

# Check Python path
python -c "import route_analyzer; print('Package OK')"
```

**GUI won't start:**
```bash
# Check GUI dependencies
pip install -r requirements_gui.txt

# Verify Streamlit installation
python -c "import streamlit; print('Streamlit OK')"

# Launch with explicit path
streamlit run project/ra_gui.py
```

**No trajectories loaded:**
- Check file paths and glob patterns
- Verify column mappings match your CSV headers
- Ensure files contain X, Z coordinates (Time optional)
- Check scale factor - try `--scale 1.0` for raw coordinates

**Clustering issues:**
- Increase `--k` if too few clusters found
- Try `--cluster_method auto` for automatic cluster detection
- Adjust `--min_samples` for DBSCAN clustering
- Check `--angle_eps` for angle-based clustering

**Performance issues:**
- Reduce number of trajectories for testing
- Use sample data for initial setup
- Close other browser tabs when using GUI
- Consider using `--decision_mode pathlen` for faster analysis

**Memory errors:**
- Process data in smaller batches
- Reduce `--distance` parameter
- Use `--scale` to reduce coordinate precision

### Getting Help

1. **Check console output** for detailed error messages
2. **Verify data format** - CSV files should have X, Z columns
3. **Try sample data first** to test installation
4. **Check file permissions** for output directories
5. **Review configuration** - use `--config` for complex setups

### Data Format Requirements

**Minimum CSV columns:**
- X coordinate (any column name)
- Z coordinate (any column name)
- Time (optional, enables timing metrics)

**Example CSV:**
```csv
Time,X,Z
0.0,100.0,200.0
0.1,101.0,201.0
...
```

**Column mapping:**
```bash
--columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time
```

## Tips

- Use `--no-plot_intercepts` and `--no-show_paths` to disable plotting
- The tool prints a suggested `--epsilon` based on step statistics
- For Parquet inputs, install the `[parquet]` extra
- Use `--config` for complex multi-command setups
- Try `--cluster_method auto` for automatic cluster detection

## License

MIT
