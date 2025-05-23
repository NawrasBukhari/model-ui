import argparse
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import cv2
import torch
from ultralytics import YOLO


class DetectionTracker:
    """Tracks and manages detection statistics"""

    def __init__(self):
        self.total_frames: int = 0
        self.detection_frames: int = 0
        self.detections_by_class: Dict[str, int] = {}
        self.last_detection_frame: int = 0
        self.consecutive_detections: int = 0
        self.first_detection_time: Optional[datetime] = None

    def add_detection(self, frame_num: int, class_name: str) -> None:
        """Record a detection"""
        self.detection_frames += 1
        self.last_detection_frame = frame_num

        if class_name not in self.detections_by_class:
            self.detections_by_class[class_name] = 0
        self.detections_by_class[class_name] += 1

        if self.first_detection_time is None:
            self.first_detection_time = datetime.now()

    def get_summary(self) -> str:
        """Get detection summary stats"""
        if self.total_frames == 0:
            return "No frames processed"

        detection_percentage = (self.detection_frames / self.total_frames) * 100

        summary = [
            f"Total frames: {self.total_frames}",
            f"Frames with detections: {self.detection_frames} ({detection_percentage:.1f}%)",
            f"Detection counts by class: {self.detections_by_class}"
        ]

        if self.first_detection_time:
            summary.append(f"First detection at: {self.first_detection_time.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(summary)


def get_available_device() -> Tuple[str, str]:
    """
    Determine the best available computing device (GPU or CPU)

    Returns:
        Tuple[str, str]: Device type and description
    """
    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        return device_type, f"CUDA GPU ({device_name})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    else:
        return "cpu", "CPU (No GPU available)"


def is_iphone_video(input_path: str) -> bool:
    """
    Detect if the video is from an iPhone based on metadata and characteristics
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        _, ext = os.path.splitext(input_path.lower())

        cap.release()

        is_iphone = (
                (abs(fps - 120) < 1 or abs(fps - 60) < 1 or abs(fps - 30) < 1) and
                (width >= 1920 or height >= 1080) and
                ext in ['.mov', '.mp4']
        )

        return is_iphone
    except Exception as e:
        print(f"Error checking if video is iPhone: {e}")
        return False


def process_video(
        input_path: str,
        output_path: str,
        model_path: str,
        conf_threshold: float = 0.10,
        save_frames: bool = False,
        frames_dir: Optional[str] = None,
        batch_size: int = 4,
        output_fps: int = 30,
        scale_factor: Optional[float] = None,
        keep_original_size: bool = False,
        device: Optional[str] = None
) -> bool:
    """
    Process a video file with fire/smoke detection, optimized for iPhone 16 Pro Max videos.

    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold for detection (0-1)
        save_frames: Save frames with detections as images
        frames_dir: Directory to save detection frames
        batch_size: Batch size for processing
        output_fps: Target output FPS
        scale_factor: Optional manual scale factor for resolution (0-1)
        keep_original_size: Keep the original video resolution
        device: Computing device to use ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        bool: True if processing was successful, False otherwise
    """

    if device is None:
        device_type, device_description = get_available_device()
    else:
        device_type = device
        device_description = f"User-specified: {device}"

    print(f"Using device: {device_description}")

    print(f"Loading model from {model_path}...")
    try:

        model = YOLO(model_path)
        model.to(device_type)
        print(f"Model loaded successfully on {device_description}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    is_iphone = is_iphone_video(input_path)
    if is_iphone:
        print("Detected iPhone video - optimizing processing parameters")

    print(f"Opening input video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if is_iphone and total_frames <= 0:

        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / input_fps
        total_frames = int(duration * input_fps)
        if total_frames <= 0:
            total_frames = 9999
            print("Warning: Could not determine frame count, will process until end of video")

    print(f"Original video properties: {original_width}x{original_height}, {input_fps} FPS, {total_frames} frames")

    if keep_original_size:

        target_width = original_width
        target_height = original_height
        print("Keeping original video resolution")
    else:

        if scale_factor is not None:

            if scale_factor > 1:
                scale_factor = scale_factor / 100.0

            scale_factor = max(0.1, min(1.0, scale_factor))
        else:

            if device_type == "cuda":

                if is_iphone and original_width >= 3840:
                    scale_factor = 0.75
                elif is_iphone and original_width >= 1920:
                    scale_factor = 0.85
                else:
                    scale_factor = 1.0
            else:

                if is_iphone and original_width >= 3840:
                    scale_factor = 0.4
                elif is_iphone and original_width >= 1920:
                    scale_factor = 0.6
                else:
                    scale_factor = 0.8

        target_width = max(640, int(original_width * scale_factor) // 2 * 2)
        target_height = max(480, int(original_height * scale_factor) // 2 * 2)

        max_dimension = 3840
        if target_width > max_dimension or target_height > max_dimension:
            ratio = min(max_dimension / target_width, max_dimension / target_height)
            target_width = int(target_width * ratio) // 2 * 2
            target_height = int(target_height * ratio) // 2 * 2
            print(f"Warning: Scaled dimensions too large, limiting to {target_width}x{target_height}")

    print(f"Processing at resolution: {target_width}x{target_height}, Output FPS: {output_fps}")

    exact_frame_interval = input_fps / output_fps
    print(f"Input FPS: {input_fps}, processing ~1 out of every {exact_frame_interval:.2f} frames")

    try:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if output_path.lower().endswith(('.mp4', '.m4v')):
            video_output = output_path
        else:
            video_output = os.path.splitext(output_path)[0] + '.mp4'

        test_writer = cv2.VideoWriter(video_output, fourcc, output_fps, (10, 10))
        if test_writer.isOpened():
            test_writer.release()
        else:
            raise Exception("mp4v codec test failed")
    except Exception:

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = os.path.splitext(output_path)[0] + '.avi'
        print("Falling back to XVID codec for compatibility")

    out = cv2.VideoWriter(video_output, fourcc, output_fps, (target_width, target_height))

    if not out.isOpened():
        print(
            f"Error: Could not create video writer for {target_width}x{target_height}. Try using a different output format.")
        cap.release()
        return False

    print(f"Writing output to: {video_output}")

    if save_frames:
        if frames_dir is None:
            frames_dir = f"detection_frames_{os.path.basename(input_path).split('.')[0]}"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Detection frames will be saved to: {frames_dir}")

    tracker = DetectionTracker()
    tracker.total_frames = total_frames

    frame_count = 0
    output_count = 0
    frame_accumulator = 0.0
    start_time = time.time()
    last_alert_time = 0
    alert_cooldown = 5

    frame_buffer: List[Any] = []
    frame_indices: List[int] = []

    if device_type == "cuda":

        batch_size = max(batch_size, 8)
        if torch.cuda.get_device_properties(0).total_memory > 8000000000:
            batch_size = max(batch_size, 16)
    elif device_type == "mps":

        batch_size = max(batch_size, 6)
    else:

        batch_size = min(batch_size, 4)

    print(f"Using optimized batch size: {batch_size}")

    if is_iphone and input_fps > 60:
        print("Using specialized iPhone 120fps video processing mode")

    print("Starting detection...")
    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:

                    if frame_buffer:
                        process_frames(model, frame_buffer, frame_indices, conf_threshold,
                                       out, tracker, save_frames, frames_dir, device_type)
                    break

                frame_count += 1

                if frame_count > tracker.total_frames:
                    tracker.total_frames = frame_count

                frame_accumulator += 1.0

                process_this_frame = False
                if frame_accumulator >= exact_frame_interval:
                    process_this_frame = True
                    frame_accumulator -= exact_frame_interval
                    output_count += 1

                if process_this_frame:

                    if is_iphone:
                        alpha = 1.1
                        beta = 5

                        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                        frame = adjusted_frame

                    if not keep_original_size and (frame.shape[1] != target_width or frame.shape[0] != target_height):
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                    else:
                        resized_frame = frame.copy()

                    frame_buffer.append(resized_frame)
                    frame_indices.append(frame_count)

                    if len(frame_buffer) >= batch_size:

                        if output_count % 10 == 0:
                            elapsed_time = time.time() - start_time
                            fps_processing = output_count / elapsed_time if elapsed_time > 0 else 0
                            progress = (frame_count / tracker.total_frames) * 100
                            remaining_frames = tracker.total_frames - frame_count
                            eta = (elapsed_time / frame_count) * remaining_frames if frame_count > 0 else 0
                            print(f"Progress: {progress:.1f}% ({frame_count}/{tracker.total_frames}) - "
                                  f"ETA: {eta:.1f}s - Processing: {fps_processing:.1f} FPS")

                        detection_found = process_frames(model, frame_buffer, frame_indices, conf_threshold,
                                                         out, tracker, save_frames, frames_dir, device_type)

                        frame_buffer = []
                        frame_indices = []

                        if detection_found and time.time() - last_alert_time > alert_cooldown:
                            print("⚠️ FIRE/SMOKE DETECTED! ⚠️")
                            last_alert_time = time.time()

        elapsed_time = time.time() - start_time
        processing_fps = output_count / elapsed_time if elapsed_time > 0 else 0

        print("\n========== Processing Summary ==========")
        print(f"Processing complete in {elapsed_time:.1f} seconds")
        print(f"Input frames: {frame_count}, Output frames: {output_count}")
        print(f"Processing speed: {processing_fps:.1f} FPS")
        print(f"Using device: {device_description}")
        print(f"\n{tracker.get_summary()}")
        print("========================================")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:

        cap.release()
        out.release()

        if os.path.exists(video_output):
            size_mb = os.path.getsize(video_output) / (1024 * 1024)
            if size_mb > 0:
                print(f"Output verified: {video_output} ({size_mb:.1f} MB)")
            else:
                print(f"Warning: Output file is empty (0 bytes)")
        else:
            print(f"Error: Output file was not created")

    return True


def process_frames(
        model: Any,
        frames: List[Any],
        frame_indices: List[int],
        conf_threshold: float,
        video_writer: cv2.VideoWriter,
        tracker: DetectionTracker,
        save_frames: bool,
        frames_dir: Optional[str],
        device: str = "cpu"
) -> bool:
    """
    Process frames and write to output video

    Args:
        model: YOLO model
        frames: List of frames to process
        frame_indices: List of corresponding frame indices
        conf_threshold: Confidence threshold for detection
        video_writer: OpenCV VideoWriter object
        tracker: DetectionTracker object
        save_frames: Whether to save detection frames
        frames_dir: Directory to save detection frames
        device: Computing device to use

    Returns:
        bool: True if any detection was found, False otherwise
    """

    results = model.predict(source=frames, conf=conf_threshold, device=device)

    any_detection = False

    for i, (r, frame_idx) in enumerate(zip(results, frame_indices)):
        frame = frames[i].copy()
        annotated_frame = frame.copy()

        detection_in_frame = False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if r.boxes is not None and len(r.boxes) > 0:

            annotated_frame = r.plot()

            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = r.names[cls_id].lower()
                conf = float(box.conf.item())

                if "fire" in label or "smoke" in label:
                    detection_in_frame = True
                    any_detection = True

                    tracker.add_detection(frame_idx, label)

                    cv2.putText(annotated_frame, f"{label.upper()} - {conf:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(annotated_frame, timestamp, (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if detection_in_frame:
            if save_frames and frames_dir is not None:
                # Save annotated frame
                frame_path = os.path.join(frames_dir, f"detection_frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, annotated_frame)

                # Save YOLO label file
                label_path = frame_path.replace('.jpg', '.txt')
                with open(label_path, 'w') as f:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        img_h, img_w = annotated_frame.shape[:2]
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h

                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Optionally save class names once
            if hasattr(model, 'names') and 'classes_written' not in globals():
                with open(os.path.join(frames_dir, 'classes.txt'), 'w') as cf:
                    for idx, name in model.names.items():
                        cf.write(f"{idx}: {name}\n")
                globals()['classes_written'] = True

        height, width = annotated_frame.shape[:2]
        progress_text = f"Frame: {frame_idx}/{tracker.total_frames}"
        cv2.putText(annotated_frame, progress_text, (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        video_writer.write(annotated_frame)

    return any_detection


def main():
    parser = argparse.ArgumentParser(description='Fire/Smoke Detection Video Processor')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path (default: processed_<input>.avi)')
    parser.add_argument('--model', '-m', required=True, help='YOLO model file path (.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.10,
                        help='Detection confidence threshold (0-1, default: 0.20)')
    parser.add_argument('--batch', '-b', type=int, default=4, help='Batch size for processing (default: 4)')
    parser.add_argument('--save-frames', '-s', action='store_true', help='Save individual frames with detections')
    parser.add_argument('--frames-dir', '-d', help='Directory to save detection frames')
    parser.add_argument('--output-fps', type=int, default=30, help='Output FPS (default: 30)')
    parser.add_argument('--scale', type=float, help='Scale factor for resolution (0-1 or percentage, default: auto)')
    parser.add_argument('--original-size', action='store_true',
                        help='Keep original video resolution (ignores --scale)')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                        help='Computing device to use (default: auto-detect)')

    args = parser.parse_args()

    if not args.output:
        output_filename = f"processed_{os.path.basename(args.input)}"
        output_path = output_filename
    else:
        output_path = args.output

    success = process_video(
        args.input,
        output_path,
        args.model,
        args.conf,
        args.save_frames,
        args.frames_dir,
        args.batch,
        args.output_fps,
        args.scale,
        args.original_size,
        args.device
    )

    return 0 if success else 1


if __name__ == "__main__":
    main()
