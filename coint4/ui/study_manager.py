#!/usr/bin/env python3
"""
Study manager for saving and loading Optuna optimization results.
"""

import json
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class StudyManager:
    """Manages Optuna study persistence and loading."""
    
    def __init__(self, studies_dir: str = "outputs/studies"):
        self.studies_dir = Path(studies_dir)
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.studies_dir / "studies_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load studies metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"studies": []}
    
    def _save_metadata(self):
        """Save studies metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_study(
        self,
        study: optuna.Study,
        description: str = "",
        tags: List[str] = None,
        config_path: str = None
    ) -> str:
        """Save study with metadata.
        
        Returns:
            Study ID for future reference
        """
        if not OPTUNA_AVAILABLE:
            return ""
        
        study_id = f"{study.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save study info
        study_info = {
            "id": study_id,
            "name": study.study_name,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "tags": tags or [],
            "config_path": config_path,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "direction": study.direction.name,
            "db_path": f"{study_id}.db"
        }
        
        # Add to metadata
        self.metadata["studies"].append(study_info)
        self._save_metadata()
        
        # Export best params
        params_file = self.studies_dir / f"{study_id}_best_params.yaml"
        with open(params_file, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        
        # Export all trials
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None
            })
        
        trials_file = self.studies_dir / f"{study_id}_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save optimization history as CSV
        history_df = pd.DataFrame([
            {
                "trial": t.number,
                "value": t.value,
                **t.params
            }
            for t in study.trials if t.value is not None
        ])
        
        if not history_df.empty:
            csv_file = self.studies_dir / f"{study_id}_history.csv"
            history_df.to_csv(csv_file, index=False)
        
        return study_id
    
    def load_study(self, study_id: str) -> Optional[optuna.Study]:
        """Load study by ID.
        
        Returns:
            Optuna Study object or None if not found
        """
        if not OPTUNA_AVAILABLE:
            return None
        
        # Find study in metadata
        study_info = None
        for s in self.metadata["studies"]:
            if s["id"] == study_id:
                study_info = s
                break
        
        if not study_info:
            return None
        
        # Load from database
        db_path = self.studies_dir / study_info["db_path"]
        if not db_path.exists():
            # Try to find by pattern
            db_files = list(self.studies_dir.glob(f"{study_info['name']}*.db"))
            if db_files:
                db_path = db_files[0]
            else:
                return None
        
        storage = f"sqlite:///{db_path}"
        
        try:
            study = optuna.load_study(
                study_name=study_info["name"],
                storage=storage
            )
            return study
        except:
            return None
    
    def list_studies(
        self,
        tags: List[str] = None,
        min_trials: int = None,
        min_value: float = None
    ) -> List[Dict[str, Any]]:
        """List available studies with optional filtering.
        
        Args:
            tags: Filter by tags
            min_trials: Minimum number of trials
            min_value: Minimum best value
        
        Returns:
            List of study metadata
        """
        studies = self.metadata["studies"]
        
        # Apply filters
        if tags:
            studies = [s for s in studies if any(t in s.get("tags", []) for t in tags)]
        
        if min_trials:
            studies = [s for s in studies if s.get("n_trials", 0) >= min_trials]
        
        if min_value is not None:
            studies = [s for s in studies if s.get("best_value", -float('inf')) >= min_value]
        
        # Sort by timestamp (newest first)
        studies.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return studies
    
    def delete_study(self, study_id: str) -> bool:
        """Delete study and its files.
        
        Returns:
            True if deleted successfully
        """
        # Find study in metadata
        study_info = None
        for i, s in enumerate(self.metadata["studies"]):
            if s["id"] == study_id:
                study_info = s
                del self.metadata["studies"][i]
                break
        
        if not study_info:
            return False
        
        # Delete files
        patterns = [
            f"{study_id}*.db",
            f"{study_id}*.yaml",
            f"{study_id}*.json",
            f"{study_id}*.csv"
        ]
        
        for pattern in patterns:
            for file in self.studies_dir.glob(pattern):
                file.unlink()
        
        # Save updated metadata
        self._save_metadata()
        
        return True
    
    def export_study(self, study_id: str, export_dir: str) -> bool:
        """Export study to a directory.
        
        Args:
            study_id: Study to export
            export_dir: Directory to export to
        
        Returns:
            True if exported successfully
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Find study files
        patterns = [
            f"{study_id}*.db",
            f"{study_id}*.yaml",
            f"{study_id}*.json",
            f"{study_id}*.csv"
        ]
        
        exported_files = []
        for pattern in patterns:
            for file in self.studies_dir.glob(pattern):
                dest = export_path / file.name
                import shutil
                shutil.copy2(file, dest)
                exported_files.append(dest)
        
        # Export metadata
        study_info = None
        for s in self.metadata["studies"]:
            if s["id"] == study_id:
                study_info = s
                break
        
        if study_info:
            metadata_file = export_path / f"{study_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(study_info, f, indent=2)
            exported_files.append(metadata_file)
        
        return len(exported_files) > 0
    
    def compare_studies(self, study_ids: List[str]) -> pd.DataFrame:
        """Compare multiple studies.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for study_id in study_ids:
            study_info = None
            for s in self.metadata["studies"]:
                if s["id"] == study_id:
                    study_info = s
                    break
            
            if study_info:
                comparison_data.append({
                    "Study ID": study_id,
                    "Timestamp": study_info.get("timestamp", "")[:19],
                    "Trials": study_info.get("n_trials", 0),
                    "Best Value": study_info.get("best_value", None),
                    "Description": study_info.get("description", "")[:50],
                    **{f"param_{k}": v for k, v in study_info.get("best_params", {}).items()}
                })
        
        if comparison_data:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame()
    
    def get_convergence_data(self, study_id: str) -> pd.DataFrame:
        """Get convergence data for plotting.
        
        Returns:
            DataFrame with trial number and best value so far
        """
        # Load trials data
        trials_file = self.studies_dir / f"{study_id}_trials.json"
        if not trials_file.exists():
            return pd.DataFrame()
        
        with open(trials_file, 'r') as f:
            trials_data = json.load(f)
        
        # Calculate convergence
        convergence = []
        best_so_far = -float('inf')
        
        for trial in trials_data:
            if trial["value"] is not None:
                best_so_far = max(best_so_far, trial["value"])
                convergence.append({
                    "trial": trial["number"],
                    "value": trial["value"],
                    "best_so_far": best_so_far
                })
        
        return pd.DataFrame(convergence)


def test_study_manager():
    """Test the study manager."""
    manager = StudyManager()
    
    # List available studies
    studies = manager.list_studies()
    print(f"Found {len(studies)} studies")
    
    for study in studies[:5]:
        print(f"- {study['id']}: {study['n_trials']} trials, best={study.get('best_value', 'N/A')}")
    
    # Test comparison
    if len(studies) >= 2:
        study_ids = [s['id'] for s in studies[:2]]
        comparison = manager.compare_studies(study_ids)
        print("\nComparison:")
        print(comparison)


if __name__ == "__main__":
    test_study_manager()