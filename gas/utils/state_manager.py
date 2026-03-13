import os
import shutil
import time
import traceback
import tempfile
from typing import Any, Dict, Optional
import numpy as np
import bittensor as bt


class StateManager:
    """
    Manages atomic state saving and loading with backup functionality.
    """
    
    def __init__(self, base_dir: str, max_backup_age_hours: float = 24.0):
        self.base_dir = base_dir
        self.max_backup_age_hours = max_backup_age_hours
        self.current_dir = os.path.join(base_dir, "state_current")
        self.backup_dir = os.path.join(base_dir, "state_backup")
        self.temp_dir = os.path.join(base_dir, "state_temp")
    
    def save_state(self, state_data: Dict[str, Any] = None, state_objects: list = None) -> bool:
        """
        Atomically save state data with backup.
        
        Args:
            state_data: Dictionary of {filename: numpy_array} for simple numpy data
            state_objects: List of tuples (object, filename) where object has save_state(save_dir, filename) method
            
        Returns:
            True if save was successful, False otherwise
        """
        temp_dir = None
        try:
            os.makedirs(self.base_dir, exist_ok=True)

            # Create a unique temp directory per save to avoid partial writes on crash
            temp_dir = tempfile.mkdtemp(dir=self.base_dir, prefix="state_temp_")
            bt.logging.trace(f"Saving state using temp dir: {temp_dir}")

            # Save simple state data (numpy arrays)
            if state_data is not None:
                for filename, value in state_data.items():
                    if isinstance(value, np.ndarray):
                        np.save(os.path.join(temp_dir, filename), value)
                        bt.logging.debug(f"Saved state for {filename}")
                    else:
                        bt.logging.warning(f"Object for {filename} is not a numpy array")

            # Save object states
            if state_objects is not None:
                for obj, filename in state_objects:
                    try:
                        obj.save_state(temp_dir, filename)
                        bt.logging.trace(f"Saved state for {filename}")
                    except Exception as e:
                        bt.logging.error(f"Failed to save state for {filename}: {e}")
                        bt.logging.error(traceback.format_exc())
                        # Continue without this object's state if it fails

            # Mark as complete
            with open(os.path.join(temp_dir, "complete"), "w") as f:
                f.write("1")

            # Atomic swap: backup current, then move temp to current
            if os.path.exists(self.current_dir):
                if os.path.exists(self.backup_dir):
                    shutil.rmtree(self.backup_dir)
                os.replace(self.current_dir, self.backup_dir)
            os.replace(temp_dir, self.current_dir)
            temp_dir = None

            bt.logging.success("Saved state successfully")
            return True

        except Exception as e:
            bt.logging.error(f"Error during state save: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False
        finally:
            # cleanup on failure (temp is moved to current on success)
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def load_state(self, state_data_keys: list = None, state_objects: list = None) -> Optional[Dict[str, Any]]:
        """
        Load state data, falling back to backup if needed.
        
        Args:
            state_data_keys: List of filenames to load as numpy arrays
            state_objects: List of tuples (object, filename) where object has load_state(save_dir, filename) method
            
        Returns:
            Dictionary of loaded numpy arrays, or None if loading failed
        """
        # Try to load current state first
        if self._can_load_state(self.current_dir):
            bt.logging.trace(f"Attempting to load current state from {self.current_dir}")
            state_data = self._load_state_from_dir(self.current_dir, state_data_keys, state_objects)
            if state_data is not None:
                bt.logging.info("Successfully loaded current state")
                return state_data
            bt.logging.warning("Failed to load current state, trying backup")
        else:
            bt.logging.warning("Current state not found or incomplete, trying backup")
        
        # Fall back to backup if needed
        if not self._can_load_state(self.backup_dir):
            bt.logging.warning("No valid state found")
            return None
        
        # Check backup age
        backup_age_hours = self._get_backup_age_hours(self.backup_dir)
        if backup_age_hours > self.max_backup_age_hours:
            bt.logging.warning(
                f"Backup is {backup_age_hours:.2f} hours old (> {self.max_backup_age_hours} hours), skipping load"
            )
            return None
        
        bt.logging.trace(
            f"Attempting to load backup state from {self.backup_dir} (age: {backup_age_hours:.2f} hours)"
        )
        
        state_data = self._load_state_from_dir(self.backup_dir, state_data_keys, state_objects)
        if state_data is not None:
            bt.logging.info(
                f"Successfully loaded backup state (age: {backup_age_hours:.2f} hours)"
            )
            return state_data
        else:
            bt.logging.error("Failed to load backup state")
            return None
    
    def _can_load_state(self, state_dir: str) -> bool:
        """Check if a state directory exists and is complete."""
        return os.path.exists(state_dir) and os.path.exists(os.path.join(state_dir, "complete"))
    
    def _load_state_from_dir(self, state_dir: str, state_data_keys: list = None, state_objects: list = None) -> Optional[Dict[str, Any]]:
        """Load state data from a directory."""
        try:
            state_data = {}
            
            # Load simple state data (numpy arrays)
            if state_data_keys is not None:
                for filename in state_data_keys:
                    file_path = os.path.join(state_dir, filename)
                    if os.path.exists(file_path):
                        state_data[filename] = np.load(file_path)
                    else:
                        bt.logging.warning(f"Expected numpy file '{filename}' not found in {state_dir}")
                        return None

            # Load object states
            if state_objects is not None:
                for obj, filename in state_objects:
                    try:
                        success = obj.load_state(state_dir, filename)
                        if not success:
                            bt.logging.warning(f"Failed to load custom object from {filename}")
                    except Exception as e:
                        bt.logging.error(f"Error loading custom object {filename}: {e}")
                        bt.logging.error(traceback.format_exc())
            
            return state_data
            
        except Exception as e:
            bt.logging.error(f"Failed to load state from {state_dir}: {e}")
            return None
    
    def _get_backup_age_hours(self, backup_dir: str) -> float:
        """Get the age of a backup in hours."""
        try:
            complete_marker = os.path.join(backup_dir, "complete")
            marker_mod_time = os.path.getmtime(complete_marker)
            return (time.time() - marker_mod_time) / 3600
        except Exception:
            return float('inf')  # Return infinity if we can't determine age


def save_validator_state(base_dir: str, state_data: Dict[str, Any] = None, state_objects: list = None, max_backup_age_hours: float = 24.0) -> bool:
    """
    Convenience function to save validator state.
    
    Args:
        base_dir: Base directory for state storage
        state_data: Dictionary of {filename: numpy_array} for simple numpy data
        state_objects: List of tuples (object, filename) where object has save_state(save_dir, filename) method
        max_backup_age_hours: Maximum age of backup to consider valid
        
    Returns:
        True if save was successful, False otherwise
    """
    state_manager = StateManager(base_dir, max_backup_age_hours)
    return state_manager.save_state(state_data, state_objects)


def load_validator_state(base_dir: str, state_data_keys: list = None, state_objects: list = None, max_backup_age_hours: float = 24.0) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load validator state.
    
    Args:
        base_dir: Base directory for state storage
        state_data_keys: List of filenames to load as numpy arrays
        state_objects: List of tuples (object, filename) where object has load_state(save_dir, filename) method
        max_backup_age_hours: Maximum age of backup to consider valid
        
    Returns:
        Dictionary of loaded numpy arrays, or None if loading failed
    """
    state_manager = StateManager(base_dir, max_backup_age_hours)
    return state_manager.load_state(state_data_keys, state_objects) 