import sys
import argparse

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Script arguments:", sys.argv)
    
    parser = argparse.ArgumentParser()
    
    # Add arguments in this specific order
    parser.add_argument('--dict_path', 
                       type=str, 
                       required=True,
                       help='Path to dictionary file')
    
    parser.add_argument('--output_file_path',
                       type=str,
                       required=True,
                       help='Path to output directory')
    
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to config file')
    
    print("\nBefore parsing - registered arguments:")
    for action in parser._actions:
        if action.dest != 'help':
            print(f"- {action.dest}: {action.option_strings}")
    
    args = parser.parse_args()
    print("\nParsed arguments:", vars(args))