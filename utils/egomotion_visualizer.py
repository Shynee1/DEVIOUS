import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from collections import deque
import os
import json
from scipy.spatial.transform import Rotation


class EgomotionVisualizer:
    """
    A class for visualizing egomotion (camera motion) data over time.
    Tracks and visualizes translation and rotation components for both predicted and ground truth poses.
    """
    
    def __init__(self, save_path=None):
        """
        Initialize the EgomotionVisualizer.
        
        Args:
            save_path (str): Path where to save visualization outputs
        """
        self.save_path = save_path if save_path else "./egomotion_outputs"
        
        # Data storage for predicted poses
        self.timestamps = []
        self.pred_translations = []  # [tx, ty, tz]
        self.pred_rotations = []     # [rx, ry, rz]
        
        # Data storage for ground truth poses
        self.gt_translations = []    # [tx, ty, tz]
        self.gt_rotations = []       # [rx, ry, rz]

        # For trajectory visualization - predicted
        self.pred_trajectory_x = []
        self.pred_trajectory_y = []
        self.pred_trajectory_z = []
        self.pred_accumulated_rotation = np.eye(3)
        self.pred_current_position = np.array([0.0, 0.0, 0.0])
        self.pred_plot_accumulated = []

        # For trajectory visualization - ground truth
        self.gt_trajectory_x = []
        self.gt_trajectory_y = []
        self.gt_trajectory_z = []
        self.gt_accumulated_rotation = np.eye(3)
        self.gt_current_position = np.array([0.0, 0.0, 0.0])
        self.gt_plot_accumulated = []
        
        # Initialize plots
        self.fig_motion = None
        self.fig_trajectory = None
        self.axes_motion = None
        self.axes_trajectory = None
        
        # Setup plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup the matplotlib figures and axes for visualization."""
        # Motion plots (translation and rotation over time)
        self.fig_motion, self.axes_motion = plt.subplots(2, 3, figsize=(15, 10))
        self.fig_motion.suptitle('Egomotion Analysis', fontsize=16)
        
    def _setup_plots(self):
        """Setup the matplotlib figures and axes for visualization."""
        # Motion plots (translation and rotation over time)
        self.fig_motion, self.axes_motion = plt.subplots(2, 3, figsize=(15, 10))
        self.fig_motion.suptitle('Egomotion Analysis - Predicted vs Ground Truth', fontsize=16)
        
        # Translation plots
        self.axes_motion[0, 0].set_title('Relative Translation X')
        self.axes_motion[0, 0].set_ylabel('tx (m)')
        self.axes_motion[0, 0].grid(True)
        
        self.axes_motion[0, 1].set_title('Relative Translation Y')
        self.axes_motion[0, 1].set_ylabel('ty (m)')
        self.axes_motion[0, 1].grid(True)
        
        self.axes_motion[0, 2].set_title('Relative Translation Z')
        self.axes_motion[0, 2].set_ylabel('tz (m)')
        self.axes_motion[0, 2].grid(True)
        
        # Rotation plots
        self.axes_motion[1, 0].set_title('Relative Rotation Z')
        self.axes_motion[1, 0].set_ylabel('rz (rad)')
        self.axes_motion[1, 0].set_xlabel('Time (s)')
        self.axes_motion[1, 0].grid(True)
        
        self.axes_motion[1, 1].set_title('Relative Rotation Y')
        self.axes_motion[1, 1].set_ylabel('ry (rad)')
        self.axes_motion[1, 1].set_xlabel('Time (s)')
        self.axes_motion[1, 1].grid(True)
        
        self.axes_motion[1, 2].set_title('Relative Rotation X')
        self.axes_motion[1, 2].set_ylabel('rx (rad)')
        self.axes_motion[1, 2].set_xlabel('Time (s)')
        self.axes_motion[1, 2].grid(True)
        
        # Trajectory plots
        self.fig_trajectory, self.axes_trajectory = plt.subplots(1, 1, figsize=(10, 10))
        self.fig_trajectory.suptitle('Camera Trajectory - Top Down View', fontsize=16)
        
        # Top-down trajectory (X vs Z, rotated by orientation)
        self.axes_trajectory.set_title('Top-Down Trajectory (Predicted vs Ground Truth)')
        self.axes_trajectory.set_xlabel('X (m)')
        self.axes_trajectory.set_ylabel('Z (m)')
        self.axes_trajectory.grid(True)
        self.axes_trajectory.set_aspect('equal')
        
    def add_measurement(self, timestamp, predicted_pose, ground_truth_pose):
        """
        Add a new egomotion measurement with both predicted and ground truth poses.
        
        Args:
            timestamp (float): Timestamp of the measurement
            predicted_pose (np.array): Predicted pose [tx, ty, tz, rx, ry, rz]
            ground_truth_pose (np.array): Ground truth pose [tx, ty, tz, rx, ry, rz]
        """

        predicted_pose = predicted_pose.cpu().numpy()
        ground_truth_pose = ground_truth_pose.cpu().numpy()

        # Extract translation and rotation from poses
        pred_translation = predicted_pose[:3]
        pred_rotation = predicted_pose[3:]
        
        gt_translation = ground_truth_pose[:3]
        gt_rotation = ground_truth_pose[3:]
        
        # Store data
        self.timestamps.append(timestamp)
        self.pred_translations.append(np.array(pred_translation))
        self.pred_rotations.append(np.array(pred_rotation))
        self.gt_translations.append(np.array(gt_translation))
        self.gt_rotations.append(np.array(gt_rotation))

        # Update predicted trajectory
        pred_rotation_matrix = Rotation.from_euler('xyz', pred_rotation, degrees=True).as_matrix()
        self.pred_accumulated_rotation = pred_rotation_matrix @ self.pred_accumulated_rotation
        self.pred_plot_accumulated.append(self.pred_accumulated_rotation.copy())
        
        pred_rotation_matrix_inv = np.linalg.inv(self.pred_accumulated_rotation)
        self.pred_current_position += pred_rotation_matrix_inv @ pred_translation
        self.pred_trajectory_x.append(self.pred_current_position[0])
        self.pred_trajectory_y.append(self.pred_current_position[1])
        self.pred_trajectory_z.append(self.pred_current_position[2])
        
        # Update ground truth trajectory
        gt_rotation_matrix = Rotation.from_euler('xyz', gt_rotation, degrees=True).as_matrix()
        self.gt_accumulated_rotation = gt_rotation_matrix @ self.gt_accumulated_rotation
        self.gt_plot_accumulated.append(self.gt_accumulated_rotation.copy())
        
        gt_rotation_matrix_inv = np.linalg.inv(self.gt_accumulated_rotation)
        self.gt_current_position += gt_rotation_matrix_inv @ gt_translation
        self.gt_trajectory_x.append(self.gt_current_position[0])
        self.gt_trajectory_y.append(self.gt_current_position[1])
        self.gt_trajectory_z.append(self.gt_current_position[2])
        
    def update_plots(self, save_plots=False):
        """
        Update the motion plots with current data.
        
        Args:
            save_plots (bool): Whether to save the plots to disk
        """
        if len(self.timestamps) < 2:
            return
            
        timestamps = np.array(self.timestamps)
        pred_translations = np.array(self.pred_translations)
        gt_translations = np.array(self.gt_translations)
        pred_trajectory_x = np.array(self.pred_trajectory_x)
        pred_trajectory_y = np.array(self.pred_trajectory_y)
        pred_trajectory_z = np.array(self.pred_trajectory_z)
        gt_trajectory_x = np.array(self.gt_trajectory_x)
        gt_trajectory_y = np.array(self.gt_trajectory_y)
        gt_trajectory_z = np.array(self.gt_trajectory_z)
        
        # Clear previous plots
        for i in range(2):
            for j in range(3):
                self.axes_motion[i, j].clear()
                
        # Reapply titles and labels
        titles = [['Relative Translation X', 'Relative Translation Y', 'Relative Translation Z'],
                 ['Relative Rotation Z', 'Relative Rotation Y', 'Relative Rotation X']]
        ylabels = [['tx (m)', 'ty (m)', 'tz (m)'],
                  ['rz (rad)', 'ry (rad)', 'rx (rad)']]
        
        for i in range(2):
            for j in range(3):
                self.axes_motion[i, j].set_title(titles[i][j])
                self.axes_motion[i, j].set_ylabel(ylabels[i][j])
                self.axes_motion[i, j].grid(True)
                if i == 1:  # Bottom row
                    self.axes_motion[i, j].set_xlabel('Time (s)')
        
        # Plot predicted relative translations
        self.axes_motion[0, 0].plot(timestamps, pred_translations[:, 0], 'b-', linewidth=2, label='tx (pred)')
        self.axes_motion[0, 1].plot(timestamps, pred_translations[:, 1], 'g-', linewidth=2, label='ty (pred)')
        self.axes_motion[0, 2].plot(timestamps, pred_translations[:, 2], 'r-', linewidth=2, label='tz (pred)')

        # Plot ground truth relative translations
        self.axes_motion[0, 0].plot(timestamps, gt_translations[:, 0], 'b--', linewidth=2, alpha=0.7, label='tx (GT)')
        self.axes_motion[0, 1].plot(timestamps, gt_translations[:, 1], 'g--', linewidth=2, alpha=0.7, label='ty (GT)')
        self.axes_motion[0, 2].plot(timestamps, gt_translations[:, 2], 'r--', linewidth=2, alpha=0.7, label='tz (GT)')

        # Plot predicted relative rotations
        pred_rotations = np.array(self.pred_rotations)
        gt_rotations = np.array(self.gt_rotations)
        
        self.axes_motion[1, 0].plot(timestamps, pred_rotations[:, 2], 'r-', linewidth=2, label='rz (pred)')
        self.axes_motion[1, 1].plot(timestamps, pred_rotations[:, 1], 'g-', linewidth=2, label='ry (pred)')
        self.axes_motion[1, 2].plot(timestamps, pred_rotations[:, 0], 'b-', linewidth=2, label='rx (pred)')

        # Plot ground truth relative rotations
        self.axes_motion[1, 0].plot(timestamps, gt_rotations[:, 2], 'r--', linewidth=2, alpha=0.7, label='rz (GT)')
        self.axes_motion[1, 1].plot(timestamps, gt_rotations[:, 1], 'g--', linewidth=2, alpha=0.7, label='ry (GT)')
        self.axes_motion[1, 2].plot(timestamps, gt_rotations[:, 0], 'b--', linewidth=2, alpha=0.7, label='rx (GT)')

        # Add legends
        for i in range(2):
            for j in range(3):
                self.axes_motion[i, j].legend()
                
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(self.save_path, exist_ok=True)
            motion_path = os.path.join(self.save_path, 'egomotion_plots.png')
            self.fig_motion.savefig(motion_path, dpi=300, bbox_inches='tight')
            print(f"Motion plots saved to: {motion_path}")
            
    def update_trajectory_plots(self, save_plots=False):
        """
        Update the trajectory plots with current data.
        
        Args:
            save_plots (bool): Whether to save the plots to disk
        """
        if len(self.pred_trajectory_x) < 2:
            return
            
        # Clear previous plot
        self.axes_trajectory.clear()
            
        # Reapply title and labels
        self.axes_trajectory.set_title('Top-Down Trajectory (Predicted vs Ground Truth)')
        self.axes_trajectory.set_xlabel('X (m)')
        self.axes_trajectory.set_ylabel('Z (m)')
        self.axes_trajectory.grid(True)
        
        # Plot predicted trajectory
        self.axes_trajectory.plot(self.pred_trajectory_x, self.pred_trajectory_z, 'b-', linewidth=2, alpha=0.7, label='Predicted Trajectory')
        self.axes_trajectory.plot(self.pred_trajectory_x[-1], self.pred_trajectory_z[-1], 'bo', markersize=8, label='Current (Pred)')
        
        # Plot ground truth trajectory
        self.axes_trajectory.plot(self.gt_trajectory_x, self.gt_trajectory_z, 'r--', linewidth=2, alpha=0.7, label='Ground Truth Trajectory')
        self.axes_trajectory.plot(self.gt_trajectory_x[-1], self.gt_trajectory_z[-1], 'ro', markersize=8, label='Current (GT)')
        
        # Plot start position
        self.axes_trajectory.plot(self.pred_trajectory_x[0], self.pred_trajectory_z[0], 'go', markersize=8, label='Start')

        # Add arrow to show current predicted orientation
        if len(self.pred_trajectory_x) > 1:
            dx = self.pred_trajectory_x[-1] - self.pred_trajectory_x[-2]
            dz = self.pred_trajectory_z[-1] - self.pred_trajectory_z[-2]
            if np.sqrt(dx**2 + dz**2) > 0:
                self.axes_trajectory.arrow(self.pred_trajectory_x[-1], self.pred_trajectory_z[-1], dx*0.1, dz*0.1, 
                                         head_width=0.02, head_length=0.03, fc='blue', ec='blue')
        
        # Add arrow for ground truth orientation
        if len(self.gt_trajectory_x) > 1:
            gt_dx = self.gt_trajectory_x[-1] - self.gt_trajectory_x[-2]
            gt_dz = self.gt_trajectory_z[-1] - self.gt_trajectory_z[-2]
            if np.sqrt(gt_dx**2 + gt_dz**2) > 0:
                self.axes_trajectory.arrow(self.gt_trajectory_x[-1], self.gt_trajectory_z[-1], gt_dx*0.1, gt_dz*0.1, 
                                         head_width=0.02, head_length=0.03, fc='red', ec='red')
        
        # Add legend and set aspect ratio
        self.axes_trajectory.legend()
        self.axes_trajectory.set_aspect('equal')
            
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(self.save_path, exist_ok=True)
            trajectory_path = os.path.join(self.save_path, 'trajectory_plots.png')
            self.fig_trajectory.savefig(trajectory_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plots saved to: {trajectory_path}")
            
    def save_data(self):
        """Save the egomotion data to JSON files."""
        if len(self.timestamps) == 0:
            return
            
        # Prepare data for saving
        data = {
            'timestamps': list(self.timestamps),
            'predicted': {
                'translations': [t.tolist() for t in self.pred_translations],
                'rotations': [r.tolist() for r in self.pred_rotations],
                'trajectory': {
                    'x': list(self.pred_trajectory_x),
                    'y': list(self.pred_trajectory_y),
                    'z': list(self.pred_trajectory_z)
                }
            },
            'ground_truth': {
                'translations': [t.tolist() for t in self.gt_translations],
                'rotations': [r.tolist() for r in self.gt_rotations],
                'trajectory': {
                    'x': list(self.gt_trajectory_x),
                    'y': list(self.gt_trajectory_y),
                    'z': list(self.gt_trajectory_z)
                }
            }
        }
        
        # Save to JSON file
        data_path = os.path.join(self.save_path, 'egomotion_data.json')
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Egomotion data saved to: {data_path}")
        
    def show_plots(self):
        """Display the plots interactively."""
        plt.figure(self.fig_motion.number)
        plt.show()
        plt.figure(self.fig_trajectory.number)
        plt.show()
        
    def get_statistics(self):
        """
        Get statistical summary of the egomotion data.
        
        Returns:
            dict: Dictionary containing statistical information
        """
        if len(self.pred_translations) == 0:
            return {}
            
        pred_translations = np.array(self.pred_translations)
        pred_rotations = np.array(self.pred_rotations)
        gt_translations = np.array(self.gt_translations)
        gt_rotations = np.array(self.gt_rotations)
        
        stats = {
            'num_measurements': len(self.timestamps),
            'time_span': float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0,
            'predicted': {
                'translation_stats': {
                    'mean': pred_translations.mean(axis=0).tolist(),
                    'std': pred_translations.std(axis=0).tolist(),
                    'min': pred_translations.min(axis=0).tolist(),
                    'max': pred_translations.max(axis=0).tolist()
                },
                'rotation_stats': {
                    'mean': pred_rotations.mean(axis=0).tolist(),
                    'std': pred_rotations.std(axis=0).tolist(),
                    'min': pred_rotations.min(axis=0).tolist(),
                    'max': pred_rotations.max(axis=0).tolist()
                },
                'trajectory_stats': {
                    'total_distance': float(np.linalg.norm(self.pred_current_position)),
                    'final_position': self.pred_current_position.tolist()
                }
            },
            'ground_truth': {
                'translation_stats': {
                    'mean': gt_translations.mean(axis=0).tolist(),
                    'std': gt_translations.std(axis=0).tolist(),
                    'min': gt_translations.min(axis=0).tolist(),
                    'max': gt_translations.max(axis=0).tolist()
                },
                'rotation_stats': {
                    'mean': gt_rotations.mean(axis=0).tolist(),
                    'std': gt_rotations.std(axis=0).tolist(),
                    'min': gt_rotations.min(axis=0).tolist(),
                    'max': gt_rotations.max(axis=0).tolist()
                },
                'trajectory_stats': {
                    'total_distance': float(np.linalg.norm(self.gt_current_position)),
                    'final_position': self.gt_current_position.tolist()
                }
            }
        }
        
        # Add error statistics between predicted and ground truth
        translation_errors = np.linalg.norm(pred_translations - gt_translations, axis=1)
        rotation_errors = np.linalg.norm(pred_rotations - gt_rotations, axis=1)
        
        # Calculate trajectory errors
        pred_traj = np.array([self.pred_trajectory_x, self.pred_trajectory_y, self.pred_trajectory_z]).T
        gt_traj = np.array([self.gt_trajectory_x, self.gt_trajectory_y, self.gt_trajectory_z]).T
        trajectory_errors = np.linalg.norm(pred_traj - gt_traj, axis=1)
        
        stats['error_stats'] = {
            'translation_rmse': float(np.sqrt(np.mean(translation_errors**2))),
            'translation_mae': float(np.mean(translation_errors)),
            'rotation_rmse': float(np.sqrt(np.mean(rotation_errors**2))),
            'rotation_mae': float(np.mean(rotation_errors)),
            'trajectory_rmse': float(np.sqrt(np.mean(trajectory_errors**2))),
            'trajectory_mae': float(np.mean(trajectory_errors)),
            'final_position_error': float(np.linalg.norm(self.pred_current_position - self.gt_current_position))
        }
        
        return stats
        
    def clear_data(self):
        """Clear all stored data and reset the visualizer."""
        self.timestamps.clear()
        self.pred_translations.clear()
        self.pred_rotations.clear()
        self.gt_translations.clear()
        self.gt_rotations.clear()
        
        self.pred_trajectory_x.clear()
        self.pred_trajectory_y.clear()
        self.pred_trajectory_z.clear()
        self.gt_trajectory_x.clear()
        self.gt_trajectory_y.clear()
        self.gt_trajectory_z.clear()
        
        self.pred_current_position = np.array([0.0, 0.0, 0.0])
        self.gt_current_position = np.array([0.0, 0.0, 0.0])
        self.pred_accumulated_rotation = np.eye(3)
        self.gt_accumulated_rotation = np.eye(3)
        self.pred_plot_accumulated.clear()
        self.gt_plot_accumulated.clear()
        
        # Clear plots
        for i in range(2):
            for j in range(3):
                self.axes_motion[i, j].clear()
        self.axes_trajectory.clear()
            
        self._setup_plots()
