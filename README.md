# Massachusetts Building Analysis Dashboard

An interactive web-based visualization dashboard for analyzing Massachusetts building inventory data from the NSI-Enhanced USA Structures Dataset. This comprehensive tool provides multi-dimensional analysis of 2.09M+ buildings with advanced clustering, temporal patterns, and geospatial visualizations.

## Live Demo

[View Live Dashboard](https://samueeelsiu.github.io/MA-building-stock-old/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Data Structure](#data-structure)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Support](#support)

## Overview

This dashboard visualizes and analyzes the complete Massachusetts building inventory, integrating data from multiple authoritative sources:
- **USA Structures**: 2,091,488 building footprints
- **National Structure Inventory (NSI)**: Detailed building characteristics(building materials, built year and foudnation materials, etc.)
- **Web Soil Survey**: Soil-related details
- **Boston Permit Dataset**: Demolition records(Boston only)

The final dataset contains **63 columns** of building attributes

## Key Features

### 10 Interactive Analysis Sections

#### 1. **Overview Dashboard**
- Real-time statistics for 1.68M cleaned buildings
- Interactive occupancy distribution visualizations
- Construction timeline from pre-1940 to 2024
- Multi-level hierarchical Sankey diagrams
- A Sampled 75,000-point interactive map with dynamic filtering

#### 2. **Data Pipelines & Processing**
- Visual representation of 4-source data integration
- Data cleaning funnel (2.09M → 1.68M buildings)
- Unclassified building reclassification algorithm
- NSI point-to-polygon spatial join visualization

#### 3. **Clustering Analysis**
- K-means clustering (K=2 to 9)
- Real pre-computed clusters on full dataset
- Elbow method optimization
- Interactive 3D scatter plots
- Cluster statistics and treemap visualizations

#### 4. **Temporal Distribution**
- Annual construction patterns analysis
- 4 visualization modes (Stacked, Line, Normalized, Cumulative)
- Building type filters (All/Residential/Non-Residential)
- Total floor area trends over time

#### 5. **Pre-1940 Historic Buildings**
- 357,200 historic buildings analysis
- Occupancy class distribution
- Preservation insights
- Area-weighted statistics

#### 6. **Post-1940 Modern Construction**
- Decade-by-decade patterns
- Annual construction tracking
- Occupancy evolution analysis
- Normalized percentage views

#### 7. **Multi-Dimensional Occupancy Clustering**
- 4D to 6D dynamic clustering
- Feature selection (Material/Foundation)
- Balanced vs Random sampling (up to 25,000 points)
- Pre-computed clustering for all combinations

#### 8. **Materials & Foundation Analysis**
- Interactive correlation heatmaps
- Click-through occupancy breakdowns
- Material usage evolution (1940-2024)
- Count and area-based visualizations

#### 9. **Soil Properties & Risk Assessment**
- Drainage class analysis
- Water table depth distribution
- Engineering property evaluation
- A sampled 50,000-point risk mapping
- High-risk building identification

#### 10. **Boston Historic Shoreline**
- Buildings on land reclaimed since 1630
- Interactive historic map overlay
- Filled land construction patterns
- Material/foundation analysis on reclaimed areas

## Data Pipeline

### Processing Stages

```
Stage 1: Spatial Join Enhancement
├── Input: 2,091,488 USA Structures + 2,095,529 NSI Points
├── Process: Multi-stage intelligent matching
│   ├── Strategy 1: Single-family one-to-one matching
│   ├── Strategy 2: Multi-unit aggregation
│   └── Strategy 3: 5-meter buffer nearest neighbor
└── Output: 1,686,451 matched buildings (80.63% match rate)

Stage 1.5: Unclassified Resolution
├── Input: Buildings with OCC_DICT voting data
├── Process: Majority voting with tie-breaking
└── Output: Reclassified occupancy categories

Stage 2: Soil Data Integration
├── Input: Web Soil Survey (3 source files)
├── Process: Double-filtering for dominant components
└── Output: 12 soil property columns

Stage 3: Demolition Data
├── Input: Boston Approved Permits
├── Process: 30-meter radius spatial join
└── Output: 1,236 buildings with demolition data
```

## Installation

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-username/MA-building-stock.git
cd MA-building-stock
```

2. **File Structure Required**
```
MA-building-stock/
├── index.html                              # Main dashboard
├── building_data.json                      # Core dataset
├── building_data_samples_random_*.json     # Random samples (15 files)
├── building_data_samples_balanced_*.json   # Balanced samples (4 files)
├── historic_shoreline_buildings.json       # Boston shoreline data
├── boston_shoreline_1630.png              # Historic map image
└── README.md                               # This file
```

3. **Launch the Dashboard**

Option A: Direct website opening
```bash
# Open https://samueeelsiu.github.io/MA-building-stock/
```

Option B: Local server (recommended)
```bash
# Python 3
python -m http.server 8000

# Node.js
npx http-server

# Then navigate to http://localhost:8000
```

### Data Processing (Optional)

If you want to regenerate the data files:

```bash

# Need to contact for requesting the required CSV file

# Install dependencies
pip install pandas numpy scikit-learn geopandas matplotlib

# Run the main processor
python data_preprocessor.py

# Process historic shoreline data
python process_shoreline.py
```

## Usage Guide

### Navigation

1. Use the top navigation tabs to switch between analysis sections
2. Each section has its own control panel for filtering and customization
3. Hover over any data point for detailed information
4. Click on charts for interactive features

### Key Interactions

- **Map Controls**: Zoom with scroll, pan with drag, filter with dropdowns
- **3D Plots**: Rotate with mouse, zoom with scroll
- **Sankey Diagrams**: Click nodes for details, drag to reposition
- **Heatmaps**: Click cells for occupancy breakdown
- **Export**: Use export buttons for PNG/JSON downloads

## Data Structure

### Main Dataset Schema

```javascript
{
  "metadata": {
    "total_buildings": 1686451,
    "version": "3.2",
    "samples_files": [...],      // References to chunk files
    "date_processed": ...
  },
  "summary_stats": {
    "total_buildings": 1686451,
    "avg_year_built": 1962,
    "avg_area_sqm": 346,
    "occupancy_classes": [9 categories]
  },
  "hierarchical_distribution": {...},   // Sankey data
  "year_occ_flow": {...},               // Year→Occ→Material→Foundation→Soil
  "temporal_data": [...],               // Time series
  "clustering": {...},                  // K-means results
  "soil_analysis": {...},               // Geotechnical data
  "data_flow_stats": {...}              // Pipeline metrics
}
```

### Building Attributes (Key Columns)

- **Identification**: BUILD_ID
- **Location**: LONGITUDE, LATITUDE, PROP_ADDR, PROP_CITY
- **Physical**: HEIGHT, SQMETERS, Est GFA sqmeters
- **Classification**: OCC_CLS, PRIM_OCC, MIX_SC
- **Construction**: year_built, material_type, foundation_type
- **Soil**: drainagecl, wtdepannmin, flodfreqcl, compname

## Technologies

### Frontend
- **HTML5/CSS3**: Responsive design with modern/professional themes
- **JavaScript ES6+**: Dynamic interactions and data processing
- **Plotly.js v2.27.0**: Advanced interactive visualizations

### Backend Processing
- **Python 3.x**
  - pandas: Data manipulation
  - scikit-learn: K-means clustering
  - geopandas: Spatial operations
  - numpy: Numerical computations

### Data Formats
- **JSON**: Primary data exchange format
- **CSV**: Source data files
- **Shapefile**: Geospatial boundaries

## Performance Notes

### Optimization Strategies

1. **Data Chunking**: Split into 30 files to handle GitHub's 25MB limit
2. **Pre-computed Clustering**: All clustering results pre-calculated
3. **Sampling**: Balanced and random samples for visualization
4. **Progressive Loading**: Lazy loading of sample chunks


### Known Limitations

- Maximum 75,000 points displayed on maps simultaneously
- Some years (2006, 2009-2011, 2013-2016) have incomplete data
- Soil data coverage: 11,385 buildings lack soil information
- Real-time clustering limited to sample data


## Credits

### Development Team
- **Developer**: Lang (Samuel) Shao
- **Supervisor**: Prof. Demi Fang
- **Institution**: [Northeastern University](https://www.northeastern.edu/)
- **Lab**: [Structural Futures Lab](https://structural-futures.org/)

### Data Sources
- USA Structures Dataset
- National Structure Inventory (NSI)
- Web Soil Survey
- Boston Permits

## Support

For issues, questions, or suggestions regarding this dashboard, please contact: shao.la@northeastern.edu




