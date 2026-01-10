"""
Менеджер для управления Optuna исследованиями и их версионированием.
Позволяет отслеживать историю оптимизаций и сравнивать результаты.
"""

import os
import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import optuna
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StudyManager:
    """Менеджер для управления Optuna исследованиями."""
    
    def __init__(self, base_dir: str = "outputs/studies"):
        """
        Инициализация менеджера.
        
        Args:
            base_dir: Базовая директория для хранения исследований
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.base_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "studies_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Загружает метаданные исследований."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"studies": {}, "version": "1.0"}
    
    def _save_metadata(self):
        """Сохраняет метаданные."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def register_study(self, 
                       study_name: str,
                       db_path: str,
                       config_path: str,
                       search_space_path: str,
                       description: str = "",
                       tags: List[str] = None) -> str:
        """
        Регистрирует новое исследование в системе версионирования.
        
        Args:
            study_name: Имя исследования
            db_path: Путь к базе данных
            config_path: Путь к конфигурации
            search_space_path: Путь к пространству поиска
            description: Описание исследования
            tags: Теги для категоризации
            
        Returns:
            Уникальный ID исследования
        """
        study_id = f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata["studies"][study_id] = {
            "name": study_name,
            "db_path": str(db_path),
            "config_path": str(config_path),
            "search_space_path": str(search_space_path),
            "created_at": datetime.now().isoformat(),
            "description": description,
            "tags": tags or [],
            "status": "active"
        }
        
        self._save_metadata()
        logger.info(f"✅ Зарегистрировано исследование: {study_id}")
        return study_id
    
    def list_studies(self, 
                     status: str = None,
                     tags: List[str] = None,
                     last_n: int = None) -> pd.DataFrame:
        """
        Возвращает список исследований.
        
        Args:
            status: Фильтр по статусу (active, archived, failed)
            tags: Фильтр по тегам
            last_n: Показать только N последних
            
        Returns:
            DataFrame с информацией об исследованиях
        """
        studies_list = []
        
        for study_id, info in self.metadata["studies"].items():
            # Фильтрация по статусу
            if status and info.get("status") != status:
                continue
            
            # Фильтрация по тегам
            if tags and not any(tag in info.get("tags", []) for tag in tags):
                continue
            
            # Проверяем существование файла
            db_exists = Path(info["db_path"]).exists()
            
            # Получаем статистику если файл существует
            stats = self._get_study_stats(info["db_path"]) if db_exists else {}
            
            studies_list.append({
                "id": study_id,
                "name": info["name"],
                "created": info["created_at"],
                "status": info["status"],
                "db_exists": db_exists,
                "trials": stats.get("trials", 0),
                "best_value": stats.get("best_value"),
                "best_sharpe": stats.get("best_sharpe"),
                "description": info.get("description", ""),
                "tags": ", ".join(info.get("tags", []))
            })
        
        df = pd.DataFrame(studies_list)
        
        if not df.empty:
            df = df.sort_values("created", ascending=False)
            if last_n:
                df = df.head(last_n)
        
        return df
    
    def _get_study_stats(self, db_path: str) -> Dict[str, Any]:
        """Получает статистику исследования из базы данных."""
        stats = {}
        
        try:
            # Подключаемся к SQLite напрямую для быстрого доступа
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Получаем количество trials
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'")
            stats["trials"] = cursor.fetchone()[0]
            
            # Получаем лучшее значение
            cursor.execute("SELECT MAX(value) FROM trials WHERE state = 'COMPLETE'")
            best_value = cursor.fetchone()[0]
            stats["best_value"] = best_value
            
            # Пытаемся получить Sharpe из user_attrs
            cursor.execute("""
                SELECT key, value_json 
                FROM trial_user_attributes 
                WHERE key = 'metrics' 
                AND trial_id = (
                    SELECT trial_id FROM trials 
                    WHERE state = 'COMPLETE' 
                    ORDER BY value DESC LIMIT 1
                )
            """)
            
            result = cursor.fetchone()
            if result:
                try:
                    metrics = json.loads(result[1])
                    stats["best_sharpe"] = metrics.get("sharpe")
                except:
                    pass
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Не удалось получить статистику для {db_path}: {e}")
        
        return stats
    
    def archive_study(self, study_id: str, reason: str = ""):
        """
        Архивирует исследование.
        
        Args:
            study_id: ID исследования
            reason: Причина архивирования
        """
        if study_id not in self.metadata["studies"]:
            logger.error(f"Исследование {study_id} не найдено")
            return
        
        study_info = self.metadata["studies"][study_id]
        
        # Перемещаем файл в архив
        if Path(study_info["db_path"]).exists():
            archive_path = self.archive_dir / Path(study_info["db_path"]).name
            shutil.move(study_info["db_path"], archive_path)
            study_info["db_path"] = str(archive_path)
        
        # Обновляем метаданные
        study_info["status"] = "archived"
        study_info["archived_at"] = datetime.now().isoformat()
        study_info["archive_reason"] = reason
        
        self._save_metadata()
        logger.info(f"📦 Исследование {study_id} архивировано")
    
    def compare_studies(self, study_ids: List[str]) -> pd.DataFrame:
        """
        Сравнивает несколько исследований.
        
        Args:
            study_ids: Список ID исследований для сравнения
            
        Returns:
            DataFrame со сравнительной статистикой
        """
        comparison_data = []
        
        for study_id in study_ids:
            if study_id not in self.metadata["studies"]:
                logger.warning(f"Исследование {study_id} не найдено")
                continue
            
            info = self.metadata["studies"][study_id]
            
            if not Path(info["db_path"]).exists():
                logger.warning(f"База данных для {study_id} не найдена")
                continue
            
            stats = self._get_study_stats(info["db_path"])
            
            comparison_data.append({
                "study_id": study_id,
                "name": info["name"],
                "trials": stats.get("trials", 0),
                "best_value": stats.get("best_value"),
                "best_sharpe": stats.get("best_sharpe"),
                "config": Path(info["config_path"]).stem,
                "search_space": Path(info["search_space_path"]).stem,
                "created": info["created_at"][:10]  # Только дата
            })
        
        return pd.DataFrame(comparison_data)
    
    def export_best_params(self, study_id: str, output_path: str = None) -> Dict[str, Any]:
        """
        Экспортирует лучшие параметры исследования.
        
        Args:
            study_id: ID исследования
            output_path: Путь для сохранения (опционально)
            
        Returns:
            Словарь с лучшими параметрами
        """
        if study_id not in self.metadata["studies"]:
            logger.error(f"Исследование {study_id} не найдено")
            return {}
        
        info = self.metadata["studies"][study_id]
        
        try:
            storage = f"sqlite:///{info['db_path']}"
            study = optuna.load_study(study_name=info["name"], storage=storage)
            
            best_params = {
                "study_id": study_id,
                "study_name": info["name"],
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "total_trials": len(study.trials),
                "exported_at": datetime.now().isoformat()
            }
            
            # Добавляем метрики если есть
            if hasattr(study.best_trial, 'user_attrs'):
                best_params["metrics"] = study.best_trial.user_attrs.get("metrics", {})
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(best_params, f, indent=2, default=str)
                logger.info(f"✅ Параметры экспортированы в {output_path}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Ошибка экспорта параметров: {e}")
            return {}
    
    def cleanup_old_studies(self, days: int = 30, dry_run: bool = True):
        """
        Удаляет или архивирует старые исследования.
        
        Args:
            days: Возраст в днях для архивирования
            dry_run: Если True, только показывает что будет удалено
        """
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        to_archive = []
        
        for study_id, info in self.metadata["studies"].items():
            if info["status"] != "active":
                continue
            
            created = datetime.fromisoformat(info["created_at"]).timestamp()
            
            if created < cutoff_date:
                to_archive.append(study_id)
        
        if dry_run:
            logger.info(f"🔍 Найдено {len(to_archive)} исследований старше {days} дней:")
            for study_id in to_archive:
                logger.info(f"  - {study_id}")
        else:
            for study_id in to_archive:
                self.archive_study(study_id, f"Старше {days} дней")
            logger.info(f"📦 Архивировано {len(to_archive)} исследований")
    
    def get_study_summary(self, study_id: str) -> str:
        """
        Возвращает подробную информацию об исследовании.
        
        Args:
            study_id: ID исследования
            
        Returns:
            Строка с подробной информацией
        """
        if study_id not in self.metadata["studies"]:
            return f"❌ Исследование {study_id} не найдено"
        
        info = self.metadata["studies"][study_id]
        stats = self._get_study_stats(info["db_path"]) if Path(info["db_path"]).exists() else {}
        
        summary = f"""
{'='*60}
📊 ИССЛЕДОВАНИЕ: {study_id}
{'='*60}
Имя: {info['name']}
Создано: {info['created_at']}
Статус: {info['status']}
Описание: {info.get('description', 'Нет описания')}
Теги: {', '.join(info.get('tags', []))}

Конфигурация: {info['config_path']}
Пространство поиска: {info['search_space_path']}
База данных: {info['db_path']}

Статистика:
  Trials: {stats.get('trials', 0)}
  Лучшее значение: {stats.get('best_value', 'N/A')}
  Лучший Sharpe: {stats.get('best_sharpe', 'N/A')}
{'='*60}
"""
        return summary


def create_study_manager() -> StudyManager:
    """Создает экземпляр менеджера исследований."""
    return StudyManager()