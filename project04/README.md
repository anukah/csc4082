# CSC4082 – Group Project 04

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
1. Run SIFT detector with simple 5x5 patch descriptor
2. Run SIFT detector with full 128-D SIFT descriptor
3. Run Harris corner detector with simple patch descriptor
4. Compare different parameters for both methods
5. Save all results to the `results/` directory

### Create test image pairs
To generate test image pairs from your own image:
```bash
python create_test_pairs.py your_image.jpg
```

This creates 10 different transformation pairs (translation, rotation, scale, etc.) in the `images/` directory.

## Project Structure

```
project04/
├── main.py                    # Main script
├── sift_implementation.py     # SIFT detection and matching
├── harris_implementation.py   # Harris corner detection
├── parameter_comparison.py    # Parameter comparison functions
├── utils.py                   # Utility functions
├── create_test_pairs.py       # Generate test image pairs
├── images/                    # Input images
├── results/                   # Output results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Output

Results are saved in the `results/` directory:
- `1a_*.jpg` - SIFT with simple descriptor
- `1b_*.jpg` - SIFT with full descriptor
- `2_*.jpg` - Harris corner detection
- `*_comparison.png` - Parameter comparison plots
- `sift_vs_harris.png` - Direct comparison
