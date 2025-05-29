import numpy as np
import os
import scipy.io
from pywt import cwt
from scipy.ndimage import median_filter
from scipy.signal import morlet2, resample
from skimage.transform import resize
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Configuration for preprocessing parameters"""
    target_sensors: List[str] = None
    target_series: List[str] = None
    target_image_shape: Tuple[int, int] = (224, 224)
    median_filter_size: int = 5
    zero_crossing_threshold: float = 0.01
    outlier_threshold: int = 10000
    cwt_scales: Tuple[int, int] = (1, 128)
    morlet_w: float = 6.0
    clip_range: Tuple[float, float] = (-3.0, 3.0)
    
    def __post_init__(self):
        if self.target_sensors is None:
            self.target_sensors = ["A", "B", "C"]
        if self.target_series is None:
            self.target_series = [
                "measurementSeries_B", "measurementSeries_C", 
                "measurementSeries_D", "measurementSeries_E", 
                "measurementSeries_F"
            ]


class SignalProcessor:
    """Signal processing utilities for acoustic emission data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def find_zero_crossings(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect zero crossings in signal with noise reduction
        
        Args:
            signal: Input signal array
        
        Returns:
            Array of zero crossing indices
        """
        # Apply median filter for noise reduction
        smoothed_signal = median_filter(signal, size=self.config.median_filter_size)
        
        # Find zero crossings
        crossings = np.where(
            (np.diff(np.sign(smoothed_signal)) < 0) & 
            (np.abs(smoothed_signal[:-1]) > self.config.zero_crossing_threshold)
        )[0]
        
        return crossings
    
    def apply_hanning_window(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Hanning window to signal (currently returns original signal)
        
        Args:
            signal: Input signal
        
        Returns:
            Processed signal (currently unchanged)
        """
        return signal  # Keep original signal
    
    def normalize_signal_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Normalize signal length using resampling
        
        Args:
            signal: Input signal
            target_length: Target length for resampling
        
        Returns:
            Resampled signal
        """
        return resample(signal, target_length)
    
    def cwt_to_numpy(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform Continuous Wavelet Transform
        
        Args:
            signal: Input signal
        
        Returns:
            CWT result as numpy array
        """
        scales = np.arange(self.config.cwt_scales[0], self.config.cwt_scales[1])
        cwt_result = cwt(signal, morlet2, scales, w=self.config.morlet_w)
        return np.abs(cwt_result).astype(np.float32)
    
    def cwt_to_spectrogram_image(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert signal to spectrogram image using CWT with SpecGAN preprocessing
        
        Args:
            signal: Input signal
        
        Returns:
            Spectrogram image array
        """
        # Perform CWT
        cwt_result = self.cwt_to_numpy(signal)
        
        # Apply log scale transformation
        log_cwt = np.log1p(cwt_result)
        
        # Frequency bin-wise normalization
        norm_cwt = np.empty_like(log_cwt)
        for i in range(log_cwt.shape[0]):
            row = log_cwt[i, :]
            mean_val = np.mean(row)
            std_val = np.std(row) if np.std(row) > 0 else 1.0
            
            # Normalize, clip, and rescale
            row_norm = (row - mean_val) / std_val
            row_norm = np.clip(row_norm, self.config.clip_range[0], self.config.clip_range[1])
            norm_cwt[i, :] = row_norm / abs(self.config.clip_range[1])  # Scale to [-1, 1]
        
        # Resize to target shape
        spectrogram_image = resize(
            norm_cwt, 
            self.config.target_image_shape, 
            mode='reflect', 
            anti_aliasing=True
        )
        
        return spectrogram_image.astype(np.float32)


class CycleLengthAnalyzer:
    """Analyzer for determining optimal cycle lengths"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.signal_processor = SignalProcessor(config)
    
    def calculate_fixed_length(self, cycle_lengths: List[int]) -> int:
        if not cycle_lengths:
            raise ValueError("No cycle lengths provided")
        
        mean_length = int(np.mean(cycle_lengths))
        print(f"📌 Mean Cycle Length: {mean_length}")
        return mean_length
    
    def collect_cycle_lengths(self, root_dir: str) -> List[int]:
        cycle_lengths_all = []
        
        for series_name in os.listdir(root_dir):
            series_path = os.path.join(root_dir, series_name)
            
            if not os.path.isdir(series_path):
                continue
            
            if not series_name.startswith("measurementSeries_"):
                continue
            
            print(f"Collecting cycle lengths from {series_name}...")
            
            for tightening_level in os.listdir(series_path):
                level_path = os.path.join(series_path, tightening_level)
                
                if not os.path.isdir(level_path):
                    continue
                
                for filename in os.listdir(level_path):
                    if not filename.endswith(".mat"):
                        continue
                    
                    try:
                        cycle_lengths = self._extract_cycle_lengths(
                            os.path.join(level_path, filename)
                        )
                        # Filter outliers
                        valid_lengths = [
                            c for c in cycle_lengths 
                            if c > self.config.outlier_threshold
                        ]
                        cycle_lengths_all.extend(valid_lengths)
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        print(f"Collected {len(cycle_lengths_all)} cycle lengths")
        return cycle_lengths_all
    
    def _extract_cycle_lengths(self, filepath: str) -> List[int]:
        """Extract cycle lengths from a single .mat file"""
        data = scipy.io.loadmat(filepath)
        vibrometer_data = data["D"].flatten()
        zero_crossings = self.signal_processor.find_zero_crossings(vibrometer_data)
        
        if len(zero_crossings) <= 1:
            return []
        
        cycle_lengths = [
            zero_crossings[i+1] - zero_crossings[i] 
            for i in range(len(zero_crossings) - 1)
        ]
        
        return cycle_lengths


class AEDataProcessor:
    """Main processor for acoustic emission data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.signal_processor = SignalProcessor(config)
        self.cycle_analyzer = CycleLengthAnalyzer(config)
    
    def process_file(self, 
                    filepath: str, 
                    target_sensor: str, 
                    fixed_length: int, 
                    series_name: str, 
                    tightening_level: str,
                    output_dir: str) -> Dict[str, Any]:
        """
        Process a single .mat file
        
        Args:
            filepath: Path to .mat file
            target_sensor: Target sensor ('A', 'B', 'C')
            fixed_length: Fixed length for signal normalization
            series_name: Measurement series name
            tightening_level: Tightening level
            output_dir: Output directory
        
        Returns:
            Processing results dictionary
        """
        print(f"Processing: {os.path.basename(filepath)} | "
              f"Series: {series_name} | "
              f"Tightening Level: {tightening_level} | "
              f"Sensor: {target_sensor}")
        
        # Load data
        data = scipy.io.loadmat(filepath)
        ae_data = data[target_sensor].flatten()  # AE data for processing
        vibrometer_data = data['D'].flatten()    # Vibrometer data for zero crossings
        
        # Find zero crossings
        zero_crossings = self.signal_processor.find_zero_crossings(vibrometer_data)
        
        if len(zero_crossings) <= 1:
            print(f"Warning: Insufficient zero crossings in {filepath}")
            return self._create_empty_result(filepath, series_name, tightening_level)
        
        # Extract cycles
        cycles = [
            ae_data[zero_crossings[i]:zero_crossings[i+1]] 
            for i in range(len(zero_crossings) - 1)
        ]
        
        # Filter valid cycles
        mean_length = int(np.mean([len(c) for c in cycles]))
        valid_cycles = [c for c in cycles if len(c) >= mean_length]
        
        if not valid_cycles:
            print(f"Warning: No valid cycles found in {filepath}")
            return self._create_empty_result(filepath, series_name, tightening_level)
        
        # Process cycles
        processed_data = self._process_cycles(valid_cycles, fixed_length)
        
        # Save results
        self._save_processed_data(
            processed_data, filepath, target_sensor, 
            series_name, tightening_level, output_dir
        )
        
        return {
            "filename": os.path.basename(filepath),
            "series": series_name,
            "tightening_level": tightening_level,
            "sensor": target_sensor,
            "total_cycles": len(cycles),
            "valid_cycles": len(valid_cycles),
            "output_shape": processed_data.shape
        }
    
    def _process_cycles(self, cycles: List[np.ndarray], fixed_length: int) -> np.ndarray:
        """Process cycles to create spectrogram images"""
        num_cycles = len(cycles)
        target_shape = self.config.target_image_shape
        
        # Pre-allocate output array
        processed_data = np.empty(
            (num_cycles, target_shape[0], target_shape[1]), 
            dtype=np.float32
        )
        
        # Process each cycle
        for idx, cycle in enumerate(cycles):
            # Apply windowing
            windowed_cycle = self.signal_processor.apply_hanning_window(cycle)
            
            # Normalize length
            normalized_cycle = self.signal_processor.normalize_signal_length(
                windowed_cycle, fixed_length
            )
            
            # Convert to spectrogram image
            processed_data[idx] = self.signal_processor.cwt_to_spectrogram_image(
                normalized_cycle
            )
        
        return processed_data
    
    def _save_processed_data(self, 
                           data: np.ndarray, 
                           filepath: str, 
                           target_sensor: str,
                           series_name: str, 
                           tightening_level: str, 
                           output_dir: str) -> None:
        """Save processed data to file"""
        # Create output directory structure
        series_path = os.path.join(output_dir, series_name, tightening_level)
        os.makedirs(series_path, exist_ok=True)
        
        # Generate output filename
        base_filename = os.path.basename(filepath)
        output_filename = f"{base_filename}_{target_sensor}_resized.npy"
        output_path = os.path.join(series_path, output_filename)
        
        # Save data
        np.save(output_path, data.astype(np.float32))
        
        print(f"Saved: {output_path} with shape {data.shape}")
    
    def _create_empty_result(self, filepath: str, series_name: str, tightening_level: str) -> Dict[str, Any]:
        """Create empty result dictionary for failed processing"""
        return {
            "filename": os.path.basename(filepath),
            "series": series_name,
            "tightening_level": tightening_level,
            "sensor": "unknown",
            "total_cycles": 0,
            "valid_cycles": 0,
            "output_shape": (0, 0, 0)
        }


class PreprocessingPipeline:
    """Complete preprocessing pipeline for acoustic emission data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = AEDataProcessor(config)
        self.cycle_analyzer = CycleLengthAnalyzer(config)
    
    def run(self, root_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Run the complete preprocessing pipeline
        
        Args:
            root_dir: Root directory containing measurement data
            output_dir: Output directory for processed data
        
        Returns:
            List of processing results
        """
        print("="*60)
        print("STARTING ACOUSTIC EMISSION DATA PREPROCESSING")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Collect cycle lengths and calculate fixed length
        print("\nStep 1: Analyzing cycle lengths...")
        cycle_lengths = self.cycle_analyzer.collect_cycle_lengths(root_dir)
        
        if not cycle_lengths:
            raise ValueError("No valid cycle lengths found in data")
        
        fixed_length = self.cycle_analyzer.calculate_fixed_length(cycle_lengths)
        
        # Step 2: Process all files
        print(f"\nStep 2: Processing files...")
        print(f"Target sensors: {self.config.target_sensors}")
        print(f"Target series: {self.config.target_series}")
        print(f"Fixed length: {fixed_length}")
        
        all_results = []
        
        for target_sensor in self.config.target_sensors:
            print(f"\nProcessing sensor: {target_sensor}")
            
            for target_series in self.config.target_series:
                series_path = os.path.join(root_dir, target_series)
                
                if not os.path.isdir(series_path):
                    print(f"Warning: Series directory not found: {target_series}")
                    continue
                
                results = self._process_series(
                    series_path, target_sensor, fixed_length, 
                    target_series, output_dir
                )
                all_results.extend(results)
        
        # Step 3: Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _process_series(self, 
                       series_path: str, 
                       target_sensor: str, 
                       fixed_length: int,
                       series_name: str, 
                       output_dir: str) -> List[Dict[str, Any]]:
        results = []
        
        for tightening_level in os.listdir(series_path):
            level_path = os.path.join(series_path, tightening_level)
            
            if not os.path.isdir(level_path):
                continue
            
            for filename in os.listdir(level_path):
                if not filename.endswith(".mat"):
                    continue
                
                try:
                    result = self.processor.process_file(
                        filepath=os.path.join(level_path, filename),
                        target_sensor=target_sensor,
                        fixed_length=fixed_length,
                        series_name=series_name,
                        tightening_level=tightening_level,
                        output_dir=output_dir
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        return results
    
    def _print_summary(self, results: List[Dict[str, Any]]) -> None:
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        
        total_files = len(results)
        successful_files = len([r for r in results if r["valid_cycles"] > 0])
        
        print(f"Total files processed: {total_files}")
        print(f"Successful files: {successful_files}")
        print(f"Failed files: {total_files - successful_files}")
        
        # Group by sensor and series
        by_sensor = {}
        by_series = {}
        
        for result in results:
            sensor = result["sensor"]
            series = result["series"]
            
            if sensor not in by_sensor:
                by_sensor[sensor] = 0
            if series not in by_series:
                by_series[series] = 0
            
            if result["valid_cycles"] > 0:
                by_sensor[sensor] += 1
                by_series[series] += 1
        
        print(f"\nFiles by sensor:")
        for sensor, count in by_sensor.items():
            print(f"  {sensor}: {count} files")
        
        print(f"\nFiles by series:")
        for series, count in by_series.items():
            print(f"  {series}: {count} files")
        
        # Sample results
        print(f"\nSample results:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result['filename']} | {result['series']} | "
                  f"{result['tightening_level']} | Cycles: {result['valid_cycles']}")
        
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more files")
        
        print("="*60)


def create_default_config() -> ProcessingConfig:
    """Create default preprocessing configuration"""
    return ProcessingConfig()


def main():
    """Main preprocessing function"""
    # Configuration
    config = create_default_config()
    
    # Paths
    root_dir = "./data"
    output_dir = "./data/resized"
    
    # Update paths as needed
    if not os.path.exists(root_dir):
        print(f"Error: Root directory does not exist: {root_dir}")
        print("Please update the root_dir path in the script.")
        return
    
    # Run preprocessing
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run(root_dir, output_dir)
    
    print(f"\nPreprocessing completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()