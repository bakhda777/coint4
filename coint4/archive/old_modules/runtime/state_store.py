"""State storage for live trading persistence."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StateStore:
    """Manages persistent state for live trading."""
    
    def __init__(self, state_dir: str = "artifacts/state"):
        """Initialize state store.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / "trading_state.json"
        self.stats_file = self.state_dir / "rolling_stats.parquet"
        self.positions_file = self.state_dir / "positions.json"
        
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load state from disk or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
                return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        # Create new state
        return {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_trades": 0,
            "total_pnl": 0.0,
            "status": "initialized"
        }
    
    def save_state(self):
        """Save current state to disk."""
        self.state["last_update"] = datetime.now().isoformat()
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def save_rolling_stats(self, pair: str, stats: Dict[str, np.ndarray]):
        """Save rolling statistics for a pair.
        
        Args:
            pair: Trading pair
            stats: Dictionary with 'mean', 'std', 'prices' arrays
        """
        # Convert to DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=len(stats.get('mean', [])), freq='15T'),
            'mean': stats.get('mean', []),
            'std': stats.get('std', []),
            'price_x': stats.get('price_x', []),
            'price_y': stats.get('price_y', [])
        })
        
        # Save to parquet
        stats_file = self.state_dir / f"stats_{pair.replace('/', '_')}.parquet"
        df.to_parquet(stats_file)
        logger.debug(f"Saved rolling stats for {pair} to {stats_file}")
    
    def load_rolling_stats(self, pair: str) -> Optional[Dict[str, np.ndarray]]:
        """Load rolling statistics for a pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Dictionary with statistics or None if not found
        """
        stats_file = self.state_dir / f"stats_{pair.replace('/', '_')}.parquet"
        
        if not stats_file.exists():
            return None
        
        try:
            df = pd.read_parquet(stats_file)
            
            return {
                'mean': df['mean'].values,
                'std': df['std'].values,
                'price_x': df['price_x'].values,
                'price_y': df['price_y'].values
            }
        except Exception as e:
            logger.error(f"Failed to load stats for {pair}: {e}")
            return None
    
    def save_position(self, pair: str, position: Dict[str, Any]):
        """Save position information.
        
        Args:
            pair: Trading pair
            position: Position details
        """
        positions = self._load_positions()
        positions[pair] = {
            **position,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(positions, f, indent=2)
            logger.debug(f"Saved position for {pair}")
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
    
    def load_position(self, pair: str) -> Optional[Dict[str, Any]]:
        """Load position for a pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Position details or None
        """
        positions = self._load_positions()
        return positions.get(pair)
    
    def _load_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load all positions from disk."""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load positions: {e}")
        return {}
    
    def clear_position(self, pair: str):
        """Clear position for a pair.
        
        Args:
            pair: Trading pair
        """
        positions = self._load_positions()
        if pair in positions:
            del positions[pair]
            
            try:
                with open(self.positions_file, 'w') as f:
                    json.dump(positions, f, indent=2)
                logger.debug(f"Cleared position for {pair}")
            except Exception as e:
                logger.error(f"Failed to clear position: {e}")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update trading metrics in state.
        
        Args:
            metrics: Dictionary with metrics to update
        """
        for key, value in metrics.items():
            self.state[key] = value
        self.save_state()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state.
        
        Returns:
            State summary dictionary
        """
        positions = self._load_positions()
        
        return {
            "session_id": self.state.get("session_id"),
            "uptime_hours": self._calculate_uptime(),
            "total_trades": self.state.get("total_trades", 0),
            "total_pnl": self.state.get("total_pnl", 0),
            "active_positions": len(positions),
            "status": self.state.get("status", "unknown")
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate uptime in hours."""
        try:
            start_time = datetime.fromisoformat(self.state.get("start_time"))
            uptime = datetime.now() - start_time
            return uptime.total_seconds() / 3600
        except:
            return 0.0
    
    def checkpoint(self, name: str = None):
        """Create a checkpoint of current state.
        
        Args:
            name: Optional checkpoint name
        """
        checkpoint_name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.state_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy state files
        import shutil
        
        files_to_backup = [
            self.state_file,
            self.positions_file
        ] + list(self.state_dir.glob("stats_*.parquet"))
        
        for file in files_to_backup:
            if file.exists():
                shutil.copy2(file, checkpoint_dir / file.name)
        
        logger.info(f"Created checkpoint: {checkpoint_name}")
    
    def restore_checkpoint(self, name: str):
        """Restore state from checkpoint.
        
        Args:
            name: Checkpoint name to restore
        """
        checkpoint_dir = self.state_dir / "checkpoints" / name
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {name} not found")
        
        import shutil
        
        # Restore files
        for file in checkpoint_dir.glob("*"):
            shutil.copy2(file, self.state_dir / file.name)
        
        # Reload state
        self.state = self._load_state()
        logger.info(f"Restored checkpoint: {name}")