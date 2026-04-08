import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def parse_training_log(log_path):
    """Parse the training log to extract metrics."""
    
    train_batches = []
    train_losses = []
    train_grad_norms = []
    
    val_batches = []
    val_per_avg = []
    val_per_by_day = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training batches
            train_match = re.search(r'Train batch (\d+): loss: ([\d.]+) grad norm: ([\d.]+)', line)
            if train_match:
                train_batches.append(int(train_match.group(1)))
                train_losses.append(float(train_match.group(2)))
                train_grad_norms.append(float(train_match.group(3)))
            
            # Parse validation average PER
            val_match = re.search(r'Val batch (\d+): PER \(avg\): ([\d.]+)', line)
            if val_match:
                batch = int(val_match.group(1))
                val_batches.append(batch)
                val_per_avg.append(float(val_match.group(2)))
            
            # Parse per-day validation PER
            day_match = re.search(r'(t15\.[\d.]+) val PER: ([\d.]+)', line)
            if day_match:
                day = day_match.group(1)
                per = float(day_match.group(2))
                if day not in val_per_by_day:
                    val_per_by_day[day] = {'batches': [], 'per': []}
                val_per_by_day[day]['batches'].append(val_batches[-1])
                val_per_by_day[day]['per'].append(per)
    
    return {
        'train': {
            'batches': train_batches,
            'losses': train_losses,
            'grad_norms': train_grad_norms
        },
        'val': {
            'batches': val_batches,
            'per_avg': val_per_avg,
            'per_by_day': val_per_by_day
        }
    }

def plot_training_curves(metrics, output_dir, model_name):
    """Plot training curves from parsed metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(metrics['train']['batches'], metrics['train']['losses'], alpha=0.6)
    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'{model_name} - Training Loss over Time')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient Norms
    ax = axes[0, 1]
    ax.plot(metrics['train']['batches'], metrics['train']['grad_norms'], alpha=0.6)
    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'{model_name} - Gradient Norm over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation PER (Average)
    ax = axes[1, 0]
    ax.plot(metrics['val']['batches'], metrics['val']['per_avg'], 'o-', linewidth=2)
    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Average Validation PER')
    ax.set_title(f'{model_name} - Validation Performance over Time')
    ax.grid(True, alpha=0.3)
    
    # Find and mark best PER
    best_idx = np.argmin(metrics['val']['per_avg'])
    best_batch = metrics['val']['batches'][best_idx]
    best_per = metrics['val']['per_avg'][best_idx]
    ax.plot(best_batch, best_per, 'r*', markersize=15, label=f'Best: {best_per:.4f} @ batch {best_batch}')
    ax.legend()
    
    # Plot 4: Per-day validation PER
    ax = axes[1, 1]
    for day, data in metrics['val']['per_by_day'].items():
        ax.plot(data['batches'], data['per'], alpha=0.5, label=day)
    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Validation PER')
    ax.set_title(f'{model_name} - Per-Day Validation Performance')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_dir}/training_curves.png")
    
    return fig

def process_model(model_dir, model_name):
    """Process a single model's training log."""
    log_path = os.path.join(model_dir, 'training_log')
    
    if not os.path.exists(log_path):
        print(f"Warning: Training log not found at {log_path}")
        return None
    
    print(f"\nProcessing {model_name}...")
    metrics = parse_training_log(log_path)
    fig = plot_training_curves(metrics, model_dir, model_name)
    
    # Print summary statistics
    print(f"\n{model_name} Training Summary:")
    print(f"Total training batches: {len(metrics['train']['batches'])}")
    print(f"Final training loss: {metrics['train']['losses'][-1]:.4f}")
    print(f"Best validation PER: {min(metrics['val']['per_avg']):.4f}")
    print(f"Final validation PER: {metrics['val']['per_avg'][-1]:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Base directory
    base_dir = 'trained_models'
    
    # Define models to process
    models = {
        'baseline_rnn': 'Monophones (41 classes)',
        'diphones_rnn': 'Diphones (876 classes)'
    }
    
    # If command line argument provided, process only that model
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        model_name = os.path.basename(model_dir)
        process_model(model_dir, model_name)
    else:
        # Process all models
        for model_dir, model_name in models.items():
            full_path = os.path.join(base_dir, model_dir)
            process_model(full_path, model_name)
    
    plt.show()