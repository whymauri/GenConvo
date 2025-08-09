"""
Dataset manager for GenConvoBench Q&A pairs using HuggingFace datasets.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict

from .parser import QAPair, qa_pairs_to_dataset


class GenConvoDatasetManager:
    """Manages GenConvoBench Q&A datasets using HuggingFace datasets."""
    
    def __init__(self, data_dir: str = "data/genconvo"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_qa_pairs(self, qa_pairs: List[QAPair], split_name: Optional[str] = None) -> str:
        """
        Save Q&A pairs to HuggingFace dataset.
        
        Args:
            qa_pairs: List of Q&A pairs to save
            split_name: Optional split name (e.g., 'train', 'test')
        
        Returns:
            Path to saved dataset
        """
        if not qa_pairs:
            raise ValueError("No Q&A pairs provided")
        
        # Convert to dataset format
        dataset_dict = qa_pairs_to_dataset(qa_pairs)
        
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Generate filename
        run_id = qa_pairs[0].run_id
        prompt_type = qa_pairs[0].prompt_type
        model = qa_pairs[0].model.replace('-', '_')
        
        if split_name:
            filename = f"{prompt_type}_{model}_{split_name}_{run_id}"
        else:
            filename = f"{prompt_type}_{model}_{run_id}"
        
        # Save as HuggingFace dataset directory
        output_path = self.data_dir / filename
        dataset.save_to_disk(str(output_path))
        
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
        return str(output_path)
    
    def load_qa_pairs(self, file_path: str) -> Dataset:
        """
        Load Q&A pairs from HuggingFace dataset directory.
        
        Args:
            file_path: Path to dataset directory
        
        Returns:
            HuggingFace dataset
        """
        return Dataset.load_from_disk(file_path)
    
    def list_datasets(self) -> List[str]:
        """List all available dataset directories."""
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        return [d.name for d in dataset_dirs]
    
    def load_all_datasets(self) -> DatasetDict:
        """
        Load all datasets and combine into DatasetDict.
        
        Returns:
            DatasetDict with all datasets
        """
        datasets = {}
        
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                # Extract split name from directory name
                split_name = dataset_dir.name
                datasets[split_name] = Dataset.load_from_disk(str(dataset_dir))
        
        return DatasetDict(datasets)
    
    def get_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get statistics about a dataset.
        
        Args:
            dataset: HuggingFace dataset
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_examples': len(dataset),
            'prompt_types': dataset.unique('prompt_type'),
            'models': dataset.unique('model'),
            'filenames': dataset.unique('filename'),
            'avg_question_length': sum(len(q) for q in dataset['question']) / len(dataset),
            'avg_answer_length': sum(len(a) for a in dataset['answer']) / len(dataset),
        } 