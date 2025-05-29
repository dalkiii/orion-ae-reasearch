import sys
import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocess import PreprocessingPipeline, ProcessingConfig
from src.utils.logger import setup_logging
from src.utils.utils import create_directories


def load_preprocessing_config(config_path: str) -> Dict[str, Any]:
    """Load preprocessing configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_processing_config(config_dict: Dict[str, Any]) -> ProcessingConfig:
    """Create ProcessingConfig from configuration dictionary"""
    preprocessing_config = config_dict.get('preprocessing', {})
    
    return ProcessingConfig(
        target_sensors=preprocessing_config.get('target_sensors', ["A", "B", "C"]),
        target_series=preprocessing_config.get('target_series', [
            "measurementSeries_B", "measurementSeries_C", 
            "measurementSeries_D", "measurementSeries_E", 
            "measurementSeries_F"
        ]),
        target_image_shape=tuple(preprocessing_config.get('target_image_shape', [224, 224])),
        median_filter_size=preprocessing_config.get('median_filter_size', 5),
        zero_crossing_threshold=preprocessing_config.get('zero_crossing_threshold', 0.01),
        outlier_threshold=preprocessing_config.get('outlier_threshold', 10000),
        cwt_scales=tuple(preprocessing_config.get('cwt_scales', [1, 128])),
        morlet_w=preprocessing_config.get('morlet_w', 6.0),
        clip_range=tuple(preprocessing_config.get('clip_range', [-3.0, 3.0]))
    )


def validate_paths(root_dir: str, output_dir: str) -> None:
    """Validate input and output paths"""
    if not os.path.exists(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")
    
    # Check if any measurement series exist
    found_series = False
    for item in os.listdir(root_dir):
        if item.startswith("measurementSeries_") and os.path.isdir(os.path.join(root_dir, item)):
            found_series = True
            break
    
    if not found_series:
        raise ValueError(f"No measurement series found in: {root_dir}")
    
    # Create output directory if it doesn't exist
    create_directories(output_dir)


def print_config_summary(config: ProcessingConfig, root_dir: str, output_dir: str) -> None:
    """Print configuration summary"""
    print("="*60)
    print("PREPROCESSING CONFIGURATION")
    print("="*60)
    print(f"Input directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sensors: {config.target_sensors}")
    print(f"Target series: {config.target_series}")
    print(f"Output image shape: {config.target_image_shape}")
    print(f"CWT scales: {config.cwt_scales}")
    print(f"Median filter size: {config.median_filter_size}")
    print(f"Zero crossing threshold: {config.zero_crossing_threshold}")
    print(f"Outlier threshold: {config.outlier_threshold}")
    print("="*60)


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description="Preprocess acoustic emission data")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/preprocess_config.yaml",
        help="Path to preprocessing configuration file"
    )
    parser.add_argument(
        "--root-dir", 
        type=str, 
        help="Root directory containing measurement data (overrides config)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for processed data (overrides config)"
    )
    parser.add_argument(
        "--sensors", 
        nargs="+", 
        choices=["A", "B", "C"],
        help="Target sensors to process (overrides config)"
    )
    parser.add_argument(
        "--series", 
        nargs="+",
        help="Target measurement series to process (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config_dict = load_preprocessing_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config_dict = {
            'preprocessing': {},
            'paths': {'root_dir': './data', 'output_dir': './data/resized'}
        }
    
    # Override with command line arguments
    if args.root_dir:
        config_dict.setdefault('paths', {})['root_dir'] = args.root_dir
    if args.output_dir:
        config_dict.setdefault('paths', {})['output_dir'] = args.output_dir
    if args.sensors:
        config_dict.setdefault('preprocessing', {})['target_sensors'] = args.sensors
    if args.series:
        config_dict.setdefault('preprocessing', {})['target_series'] = args.series
    
    # Get paths
    paths_config = config_dict.get('paths', {})
    root_dir = paths_config.get('root_dir', './data')
    output_dir = paths_config.get('output_dir', './data/resized')
    
    # Create processing configuration
    processing_config = create_processing_config(config_dict)
    
    # Validate paths
    validate_paths(root_dir, output_dir)
    
    # Print configuration summary
    print_config_summary(processing_config, root_dir, output_dir)
    
    # Run preprocessing
    print("\nStarting preprocessing...")
    pipeline = PreprocessingPipeline(processing_config)
    results = pipeline.run(root_dir, output_dir)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Processed {len(results)} files")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()