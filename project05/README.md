# CSC4082 – Group Project 05

S/20/369, S/20/381, S/20/139, S/20/534

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the main script
```bash
python main.py
```

This will:
1. Load the iris dataset
2. Run k-means clustering with k-means++ initialization
3. Evaluate clustering accuracy against true species labels
4. Compare random vs k-means++ initialization methods
5. Save all visualizations to the `results/` directory

## Project Structure

```
project05/
├── main.py              # Main script
├── kmeans.py            # K-means algorithm implementation
├── visualizations.py    # Visualization functions
├── data/
│   └── iris.txt         # Iris dataset
├── results/             # Output visualizations
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Output

Results are saved in the `results/` directory:
- `comparison.png` - Side-by-side clustering vs true labels
- `confusion_matrix.png` - Accuracy visualization
- `cluster_feature_pairs.png` - Feature scatter plots (clusters)
- `species_feature_pairs.png` - Feature scatter plots (species)