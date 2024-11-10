import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    print("Initial arguments:", [action.dest for action in parser._actions])
    
    # Add arguments one by one and check after each addition
    parser.add_argument('--dict_path', 
                       type=str, 
                       required=True,
                       help='Path to dictionary file')
    print("After adding dict_path:", [action.dest for action in parser._actions])
    
    parser.add_argument('--output_file_path',
                       type=str,
                       required=True,
                       help='Path to output directory')
    print("After adding output_file_path:", [action.dest for action in parser._actions])
    
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to config file')
    print("After adding config:", [action.dest for action in parser._actions])
    
    args = parser.parse_args()
    print("\nParsed args:", vars(args))



    