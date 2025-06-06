"""
Evaluation utilities for classification results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import classification_report
import os


class ResultsEvaluator:
    """Class for evaluating and analyzing classification results"""
    
    def __init__(self, class_names: Optional[List[str]] = None, results_dir: str = "./results"):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names for visualization
            results_dir: Directory to save results
        """
        self.class_names = class_names or [f"Class_{i}" for i in range(7)]
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray, 
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png") -> None:
        """
        Plot and save confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            title: Plot title
            save_name: Filename to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                #    fmt='d', 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_training_curves(self, 
                           history: Dict[str, List[float]], 
                           run_id: int = 0) -> None:
        """
        Plot training curves
        
        Args:
            history: Training history dictionary
            run_id: Run identifier
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curves (Run {run_id + 1})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
        ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Accuracy Curves (Run {run_id + 1})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.results_dir, f"training_curves_run{run_id + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
    
    def calculate_average_performance(self, 
                                   all_results: List[Dict[str, Any]], 
                                   num_classes: int = 7) -> Dict[str, Any]:
        """
        Calculate average performance across multiple runs
        
        Args:
            all_results: List of result dictionaries from multiple runs
            num_classes: Number of classes
        
        Returns:
            Dictionary containing average performance metrics
        """
        print("\n" + "="*60)
        print("AVERAGE PERFORMANCE ACROSS ALL RUNS")
        print("="*60)
        
        # Calculate average accuracy and F1
        accuracies = [result['accuracy'] for result in all_results]
        f1_scores = [result['macro_f1'] for result in all_results]
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"Average Test Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Macro F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        # Calculate class-wise averages
        avg_precision = {}
        avg_recall = {}
        avg_f1_class = {}
        
        for i in range(num_classes):
            class_str = str(i)
            precisions = [result['class_reports'][class_str]['precision'] for result in all_results]
            recalls = [result['class_reports'][class_str]['recall'] for result in all_results]
            f1s = [result['class_reports'][class_str]['f1-score'] for result in all_results]
            
            avg_precision[class_str] = np.mean(precisions)
            avg_recall[class_str] = np.mean(recalls)
            avg_f1_class[class_str] = np.mean(f1s)
        
        # Print class-wise performance
        print("\nClass-wise Average Performance:")
        print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-" * 65)
        
        for i in range(num_classes):
            class_str = str(i)
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            
            prec_mean = avg_precision[class_str]
            prec_std = np.std([result['class_reports'][class_str]['precision'] for result in all_results])
            
            rec_mean = avg_recall[class_str]
            rec_std = np.std([result['class_reports'][class_str]['recall'] for result in all_results])
            
            f1_mean = avg_f1_class[class_str]
            f1_std = np.std([result['class_reports'][class_str]['f1-score'] for result in all_results])
            
            print(f"{class_name:<15} {prec_mean:.3f} ± {prec_std:.3f}   {rec_mean:.3f} ± {rec_std:.3f}   {f1_mean:.3f} ± {f1_std:.3f}")
        
        # Calculate and plot average confusion matrix
        avg_confusion_matrix = np.mean([result['confusion_matrix'] for result in all_results], axis=0)
        self.plot_confusion_matrix(
            avg_confusion_matrix, 
            title="Average Confusion Matrix",
            save_name="average_confusion_matrix.png"
        )
        
        # Plot performance comparison across runs
        self._plot_performance_comparison(all_results, avg_accuracy, avg_f1)
        
        # Save results to file
        self._save_results_to_file(all_results, avg_accuracy, std_accuracy, avg_f1, std_f1, 
                                 avg_precision, avg_recall, avg_f1_class, num_classes)
        
        return {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_macro_f1': avg_f1,
            'std_macro_f1': std_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_class': avg_f1_class,
            'avg_confusion_matrix': avg_confusion_matrix
        }
    
    def _plot_performance_comparison(self, 
                                   all_results: List[Dict[str, Any]], 
                                   avg_accuracy: float, 
                                   avg_f1: float) -> None:
        """Plot performance comparison across runs"""
        runs = list(range(1, len(all_results) + 1))
        accuracies = [result['accuracy'] for result in all_results]
        f1_scores = [result['macro_f1'] for result in all_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(runs, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_accuracy:.4f}')
        ax1.set_xlabel('Run')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Test Accuracy by Run')
        ax1.set_xticks(runs)
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1 score comparison
        bars2 = ax2.bar(runs, f1_scores, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.axhline(y=avg_f1, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_f1:.4f}')
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Macro F1 Score')
        ax2.set_title('Macro F1 Score by Run')
        ax2.set_xticks(runs)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.results_dir, "performance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison saved to {save_path}")
    
    def _save_results_to_file(self, 
                            all_results: List[Dict[str, Any]], 
                            avg_accuracy: float, 
                            std_accuracy: float,
                            avg_f1: float, 
                            std_f1: float,
                            avg_precision: Dict[str, float],
                            avg_recall: Dict[str, float],
                            avg_f1_class: Dict[str, float],
                            num_classes: int) -> None:
        """Save detailed results to text file"""
        save_path = os.path.join(self.results_dir, "detailed_results.txt")
        
        with open(save_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("DETAILED CLASSIFICATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Test Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
            f.write(f"Average Macro F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n\n")
            
            # Individual run results
            f.write("INDIVIDUAL RUN RESULTS:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(all_results):
                f.write(f"Run {i+1}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {result['macro_f1']:.4f}\n")
            f.write("\n")
            
            # Class-wise performance
            f.write("CLASS-WISE AVERAGE PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
            f.write("-" * 65 + "\n")
            
            for i in range(num_classes):
                class_str = str(i)
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                
                prec_mean = avg_precision[class_str]
                prec_std = np.std([result['class_reports'][class_str]['precision'] for result in all_results])
                
                rec_mean = avg_recall[class_str]
                rec_std = np.std([result['class_reports'][class_str]['recall'] for result in all_results])
                
                f1_mean = avg_f1_class[class_str]
                f1_std = np.std([result['class_reports'][class_str]['f1-score'] for result in all_results])
                
                f.write(f"{class_name:<15} {prec_mean:.3f} ± {prec_std:.3f}   {rec_mean:.3f} ± {rec_std:.3f}   {f1_mean:.3f} ± {f1_std:.3f}\n")
        
        print(f"Detailed results saved to {save_path}")
    
    def plot_class_performance_heatmap(self, all_results: List[Dict[str, Any]], num_classes: int = 7) -> None:
        """Plot heatmap of class-wise performance metrics"""
        metrics = ['precision', 'recall', 'f1-score']
        data = np.zeros((num_classes, len(metrics)))
        
        for i in range(num_classes):
            class_str = str(i)
            data[i, 0] = np.mean([result['class_reports'][class_str]['precision'] for result in all_results])
            data[i, 1] = np.mean([result['class_reports'][class_str]['recall'] for result in all_results])
            data[i, 2] = np.mean([result['class_reports'][class_str]['f1-score'] for result in all_results])
        
        plt.figure(figsize=(8, 10))
        sns.heatmap(data, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   xticklabels=metrics,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Score'})
        
        plt.title('Class-wise Performance Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, "class_performance_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class performance heatmap saved to {save_path}")