# Repository Structure

```
fractalvideoguard/
│
├── README.md                          # Main documentation with quick start
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .zenodo.json                       # Zenodo metadata (NOTE: starts with dot!)
├── CITATION.cff                       # Structured citation information
│
├── fractalvideoguard_v0_5_2.py       # ⭐ Main single-file package (50 KB)
│                                      # All-in-one implementation for easy distribution
│
├── tests/
│   └── test_golden_v0_5_2.py         # Golden test suite (29 tests, 96.6% coverage)
│
├── examples/
│   ├── basic_usage.py                # Simple feature extraction example
│   ├── batch_processing.py           # Parallel batch processing
│   ├── stream_processing.py          # RTSP/webcam monitoring
│   └── custom_config.py              # Configuration customization
│
└── docs/
    ├── THEORY.md                      # 📚 FIO/QO3 theory explained (17 KB)
    ├── USAGE_GUIDE.md                 # 📖 Comprehensive usage guide (26 KB)
    ├── CONFIG_GUIDE.md                # ⚙️ Configuration reference (17 KB)
    └── AUDIT_REPORT_SINGLE_FILE_v0.5.2.md  # 🔍 Technical audit (15 KB)
```

## File Descriptions

### Core Files

#### `fractalvideoguard_v0_5_2.py`
**Main single-file package** - All functionality in one file for easy distribution.

**What's inside:**
- Configuration system (60+ parameters, 4 presets)
- Video reader (hang-proof, multi-source)
- ROI extraction (MediaPipe/Haar/fallback)
- Fractal features (DFA, box-count)
- Frequency features (DCT, FFT, ringing)
- Statistical tools (bootstrap, surrogate)
- CLI interface

**Usage:**
```bash
python fractalvideoguard_v0_5_2.py --preset fast --extract video.mp4
```

#### `tests/test_golden_v0_5_2.py`
**Comprehensive test suite** - 29 tests across 6 categories.

**Coverage:**
- Config validation (7 tests)
- Numerical stability (8 tests)
- Memory stability (4 tests)
- Rotation timeout (3 tests)
- Edge cases (5 tests)
- End-to-end pipeline (2 tests)

**Usage:**
```bash
python tests/test_golden_v0_5_2.py
```

### Documentation

#### `docs/THEORY.md`
**Theoretical foundation** - QO3/FIO Universal Attractor Theory.

**Topics:**
- Fractal-Informational Ontology
- Long-Range Dependence (LRD)
- Detrended Fluctuation Analysis (DFA)
- Box-counting dimension
- Application to deepfake detection
- Mathematical formulations
- Empirical validation

#### `docs/USAGE_GUIDE.md`
**Comprehensive usage guide** - Tutorial with examples.

**Topics:**
- Installation instructions
- Command-line interface
- Python API
- Configuration presets
- Batch processing
- Stream processing
- Custom pipelines
- Deployment
- Troubleshooting

#### `docs/CONFIG_GUIDE.md`
**Configuration reference** - All 60+ parameters documented.

**Topics:**
- Parameter descriptions
- Value ranges and defaults
- When to change parameters
- Use case examples
- Performance vs accuracy tradeoffs

#### `docs/AUDIT_REPORT_SINGLE_FILE_v0.5.2.md`
**Technical audit report** - Security and quality analysis.

**Topics:**
- Critical issues resolved
- Test coverage results
- Performance benchmarks
- Security analysis
- Deployment recommendations
- Known limitations

### Examples

#### `examples/basic_usage.py`
Simple example demonstrating:
- Feature extraction
- Result interpretation
- Deepfake scoring

#### `examples/batch_processing.py`
Batch processing example demonstrating:
- Sequential processing
- Parallel processing
- Results aggregation
- CSV export

#### `examples/stream_processing.py`
Stream processing example demonstrating:
- RTSP camera monitoring
- Webcam analysis
- Continuous loop
- Alert system

#### `examples/custom_config.py`
Configuration example demonstrating:
- Custom config creation
- Parameter tuning
- Config serialization
- Environment overrides

### Metadata Files

#### `.zenodo.json`
**Zenodo metadata** for automatic DOI assignment.

**Important:** Starts with a dot (`.zenodo.json`)!

**Contains:**
- Title and description
- Author information (ORCID)
- Keywords and subjects
- License information
- Version and date

#### `CITATION.cff`
**Citation File Format** - Structured citation metadata for automatic citation generation.

**Format:** YAML-based standard
**Used by:** GitHub, Zenodo, Zotero

#### `LICENSE`
**MIT License** - Permissive open-source license.

**Key points:**
- Free to use, modify, distribute
- Commercial use allowed
- No warranty

#### `requirements.txt`
**Python dependencies** - Minimal requirements.

**Dependencies:**
- `numpy>=1.21.0` (numerical computing)
- `opencv-python>=4.5.0` (video processing)

**Optional:**
- `mediapipe` (face detection)
- `psutil` (memory monitoring)

### Repository Root Files

#### `README.md`
**Main documentation** with:
- Overview and features
- Quick start guide
- Theoretical foundation
- Installation instructions
- Usage examples
- Performance benchmarks
- Citation information

## Usage Workflow

### 1. Quick Start (First-Time User)

```bash
# Clone and install
git clone https://github.com/yourusername/fractalvideoguard.git
cd fractalvideoguard
pip install -r requirements.txt

# Verify
python tests/test_golden_v0_5_2.py

# Analyze video
python fractalvideoguard_v0_5_2.py --preset fast --extract video.mp4
```

### 2. Learn More

Read documentation in order:
1. `README.md` - Overview
2. `docs/THEORY.md` - Understand the science
3. `docs/USAGE_GUIDE.md` - Learn all features
4. `docs/CONFIG_GUIDE.md` - Tune parameters

### 3. Integrate into Project

```python
# In your code
from fractalvideoguard_v0_5_2 import extract_features, ConfigPresets

config = ConfigPresets.production_fast()
features, debug = extract_features('video.mp4', config=config)

# Check results
if features['hurst_dfa'] < 0.60:
    print("Possible deepfake detected")
```

### 4. Contribute

1. Read `docs/AUDIT_REPORT_SINGLE_FILE_v0.5.2.md`
2. Run tests: `python tests/test_golden_v0_5_2.py`
3. Submit pull request

## For Researchers

### Citation

See `CITATION.cff` or use:

```bibtex
@software{chechelnitsky2026fractalvideoguard,
  author       = {Chechelnitsky, Igor},
  title        = {FractalVideoGuard: Deepfake Detection via 
                  Fractal-Informational Ontology},
  year         = 2026,
  version      = {0.5.2},
  doi          = {10.5281/zenodo.XXXXXX},
}
```

### Reproducibility

1. **Code:** Single file = single version
2. **Config:** JSON serialization
3. **Seeds:** Random seed control
4. **Checksums:** SHA-256 verification
5. **Tests:** Golden test suite

### Extending the Work

See `examples/` for:
- Custom feature extraction
- ML model integration
- New config presets

## For Developers

### Code Organization

Single-file design (`fractalvideoguard_v0_5_2.py`) structured as:

```python
# 1. Configuration (lines 90-440)
#    - Dataclasses for type safety
#    - Validation methods
#    - Preset factory

# 2. Video Processing (lines 441-662)
#    - VideoReader class
#    - Hang-proof rotation detection
#    - Multi-source support

# 3. ROI Extraction (lines 663-858)
#    - FaceROISource class
#    - MediaPipe/Haar/fallback
#    - ROIStandardizer (memory-efficient)

# 4. Feature Extraction (lines 859-1274)
#    - Frequency features (DCT, FFT, ringing)
#    - Fractal features (DFA, box-count)
#    - Statistical tools

# 5. CLI (lines 1450-1517)
#    - Argument parsing
#    - Preset selection
#    - Feature extraction
```

### Testing

Run specific test categories:
```bash
python tests/test_golden_v0_5_2.py --only numerical
python tests/test_golden_v0_5_2.py --only memory
python tests/test_golden_v0_5_2.py --verbose
```

### Code Quality

```bash
# Type checking
mypy fractalvideoguard_v0_5_2.py

# Linting
pylint fractalvideoguard_v0_5_2.py

# Security
bandit -r fractalvideoguard_v0_5_2.py
```

## Publishing to Zenodo

1. **GitHub Release:** Tag version (e.g., `v0.5.2`)
2. **Zenodo Integration:** Enable Zenodo-GitHub sync
3. **Automatic DOI:** Zenodo assigns DOI automatically
4. **Update References:** Replace `XXXXXX` in README with actual DOI

**Note:** `.zenodo.json` must start with a dot for Zenodo to recognize it!

## Support

- **Issues:** GitHub Issues for bugs
- **Discussions:** GitHub Discussions for questions
- **Email:** See ORCID profile for contact

---

**Last Updated:** 2026-01-18  
**Version:** 0.5.2  
**Author:** Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)
