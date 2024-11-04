import os
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import the dataset class and evaluation function from the training script
from MRCNNTraining import MaskRCNNDataset, evaluate_model, collate_fn

def calculate_iou(pred_box, gt_box):
    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)  # Removed +1 to avoid overflow

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model_debug(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """Evaluate the model with detailed debugging information."""
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    debug_info = {
        'total_images': 0,
        'empty_predictions': 0,
        'empty_ground_truth': 0,
        'successful_matches': 0,
        'failed_matches': 0,
        'low_confidence_predictions': 0
    }
    
    print("Starting evaluation with debugging...")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            debug_info['total_images'] += len(images)
            
            # Move images to device
            images = [image.to(device) for image in images]
            
            # Get model predictions
            outputs = model(images)
            
            # Process each image in the batch
            for i, (target, output) in enumerate(zip(targets, outputs)):
                print(f"\nProcessing image {debug_info['total_images']-len(images)+i+1}")
                
                # Get ground truth boxes and labels
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # Get predictions
                pred_boxes = output['boxes'].cpu()
                pred_labels = output['labels'].cpu()
                pred_scores = output['scores'].cpu()
                
                print(f"Ground truth boxes: {len(gt_boxes)}")
                print(f"Predicted boxes (before filtering): {len(pred_boxes)}")
                
                if len(gt_boxes) == 0:
                    debug_info['empty_ground_truth'] += 1
                    print("WARNING: Empty ground truth")
                    continue
                
                if len(pred_boxes) == 0:
                    debug_info['empty_predictions'] += 1
                    print("WARNING: No predictions made")
                    continue
                
                # Filter predictions by confidence threshold
                confident_mask = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[confident_mask]
                pred_labels = pred_labels[confident_mask]
                pred_scores = pred_scores[confident_mask]
                
                print(f"Predicted boxes (after confidence filtering): {len(pred_boxes)}")
                print(f"Confidence threshold: {confidence_threshold}")
                print(f"Max confidence score: {pred_scores.max().item() if len(pred_scores) > 0 else 0}")
                
                if len(pred_boxes) == 0:
                    debug_info['low_confidence_predictions'] += 1
                    print("WARNING: All predictions filtered out due to low confidence")
                    continue
                
                # Match predictions to ground truth
                matched_gt = set()
                for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gt:
                            continue
                            
                        iou = calculate_iou(pred_box, gt_box)
                        
                        print(f"Pred box {pred_idx} vs GT box {gt_idx}:")
                        print(f"  IoU: {iou:.4f}")
                        print(f"  Pred label: {pred_label.item()}")
                        print(f"  GT label: {gt_label.item()}")
                        print(f"  Confidence: {pred_score.item():.4f}")
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # If IoU is high enough, count as true positive
                    if best_iou >= iou_threshold:
                        matched_gt.add(best_gt_idx)
                        debug_info['successful_matches'] += 1
                        all_true_labels.append(gt_labels[best_gt_idx].item())
                        all_pred_labels.append(pred_label.item())
                        print(f"MATCH: Pred {pred_idx} -> GT {best_gt_idx} (IoU: {best_iou:.4f})")
                    else:
                        debug_info['failed_matches'] += 1
                        all_pred_labels.append(pred_label.item())
                        all_true_labels.append(0)  # False positive
                        print(f"NO MATCH: Pred {pred_idx} (Best IoU: {best_iou:.4f})")
                
                # Add unmatched ground truths as false negatives
                for gt_idx, gt_label in enumerate(gt_labels):
                    if gt_idx not in matched_gt:
                        all_true_labels.append(gt_label.item())
                        all_pred_labels.append(0)
                        print(f"MISSED: GT {gt_idx}")
    
    # Print debug summary
    print("\nEvaluation Debug Summary:")
    print("=" * 50)
    print(f"Total images processed: {debug_info['total_images']}")
    print(f"Images with empty ground truth: {debug_info['empty_ground_truth']}")
    print(f"Images with no predictions: {debug_info['empty_predictions']}")
    print(f"Images with only low confidence predictions: {debug_info['low_confidence_predictions']}")
    print(f"Successful matches: {debug_info['successful_matches']}")
    print(f"Failed matches: {debug_info['failed_matches']}")
    
    # Calculate metrics only if we have predictions
    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        # Convert to numpy arrays for metric calculation
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)
        
        print("\nMetrics:")
        print("=" * 50)
        
        try:
            accuracy = np.mean(all_true_labels == all_pred_labels)
            precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
            recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            
            return accuracy, precision, recall, f1, conf_matrix, debug_info
        
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return 0, 0, 0, 0, None, debug_info
    else:
        print("\nWARNING: No valid predictions to calculate metrics")
        return 0, 0, 0, 0, None, debug_info

def load_and_evaluate_model(model_path, json_file, img_dir, batch_size=1, num_workers=0):
    """Load and evaluate a saved Mask R-CNN model with debugging."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    try:
        eval_dataset = MaskRCNNDataset(json_file=json_file, img_dir=img_dir)
        print(f"Dataset loaded successfully with {len(eval_dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loader (using batch_size=1 for easier debugging)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Load the model
    try:
        # Initialize model architecture
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Load saved weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully")
        
        # Print model's number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    print("\nStarting evaluation with debugging...")
    try:
        # Try different confidence thresholds
        confidence_thresholds = [0.3, 0.5, 0.7]
        iou_thresholds = [0.3, 0.5, 0.7]
        
        best_f1 = 0
        best_threshold = None
        best_results = None
        
        for conf_thresh in confidence_thresholds:
            for iou_thresh in iou_thresholds:
                print(f"\nTrying confidence threshold: {conf_thresh}, IoU threshold: {iou_thresh}")
                results = evaluate_model_debug(
                    model=model,
                    data_loader=eval_loader,
                    device=device,
                    confidence_threshold=conf_thresh,
                    iou_threshold=iou_thresh
                )
                
                accuracy, precision, recall, f1, conf_matrix, debug_info = results
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = (conf_thresh, iou_thresh)
                    best_results = results
        
        if best_results:
            print("\nBest Results:")
            print(f"Confidence threshold: {best_threshold[0]}")
            print(f"IoU threshold: {best_threshold[1]}")
            print(f"F1 Score: {best_f1:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "MaskRCNNModelV1.2.pth"
    JSON_FILE = "annotations_in_coco.json"
    IMG_DIR = "SolDef_AI/Labeled"
    
    # Run evaluation with debugging
    load_and_evaluate_model(
        model_path=MODEL_PATH,
        json_file=JSON_FILE,
        img_dir=IMG_DIR,
        batch_size=1,  # Using batch_size=1 for easier debugging
        num_workers=0  # Using num_workers=0 for easier debugging
    )