from pathlib import Path
from typing import List, Set

def generate_tree(
    root_path: str,
    exclude_patterns: Set[str] = {'__pycache__', '.git', '.pytest_cache', '.venv', 'venv'},
    indent: str = '    ',
    prefix: str = '',
    use_emoji: bool = False
) -> str:
    """
    Generate a tree-like directory structure visualization.
   
    Args:
        root_path: Path to the root directory
        exclude_patterns: Set of patterns to exclude
        indent: Indentation string
        prefix: Prefix for current line
        use_emoji: Whether to add emoji icons before files and folders
       
    Returns:
        String representation of the directory structure
    """
    root = Path(root_path)
    if not root.exists():
        return "Directory not found"
   
    # Initialize the output with root directory
    folder_emoji = "📂 " if use_emoji else ""
    output = [f"{folder_emoji}{root.name}/"]
   
    def should_exclude(path: Path) -> bool:
        """Check if path should be excluded based on patterns."""
        return any(pattern in str(path) for pattern in exclude_patterns)
   
    def add_to_tree(directory: Path, prefix: str = '', level: int = 0) -> List[str]:
        """Recursively build tree structure."""
        contents = []
       
        # Get all items and filter out excluded ones first
        items = [item for item in directory.iterdir() if not should_exclude(item)]
        # Sort items (directories first, then alphabetically)
        items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
       
        for i, path in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = '└── ' if is_last else '├── '
            current_connector = '    ' if is_last else '│   '
           
            if path.is_dir():
                folder_prefix = "📂 " if use_emoji else ""
                contents.append(f"{prefix}{current_prefix}{folder_prefix}{path.name}/")
                contents.extend(add_to_tree(path, prefix + current_connector, level + 1))
            else:
                file_prefix = "📄 " if use_emoji else ""
                contents.append(f"{prefix}{current_prefix}{file_prefix}{path.name}")
               
        return contents
   
    output.extend(add_to_tree(root))
    return '\n'.join(output)

def main():
    """Main function to demonstrate usage."""
    root_directory = "."  # Current directory
    exclude_patterns = {'__pycache__', '.git', '.vscode', 'node_modules', 'assets'}
   
    try:
        # Generate tree without emoji
        print("Tree without emoji:")
        tree = generate_tree(root_directory, exclude_patterns, use_emoji=False)
        print(tree)
        
        print("\nTree with emoji:")
        # Generate tree with emoji
        tree_emoji = generate_tree(root_directory, exclude_patterns, use_emoji=True)
        print(tree_emoji)
    except Exception as e:
        print(f"Error generating directory structure: {e}")

if __name__ == "__main__":
    main()
