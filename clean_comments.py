#!/usr/bin/env python3
"""Remove commented lines from omega4_main.py while preserving functionality"""

import re

def clean_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        # Keep the shebang
        if i == 0 and line.startswith('#!'):
            cleaned_lines.append(line)
            continue
        
        # Handle docstrings
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in line else "'''"
            else:
                if docstring_char in line:
                    in_docstring = False
            cleaned_lines.append(line)
            continue
        
        if in_docstring:
            cleaned_lines.append(line)
            continue
        
        # Skip lines that are entirely comments (but not in docstrings)
        stripped = line.strip()
        if stripped.startswith('#'):
            # Keep import-related comments that might be important
            if 'PHASE' in line or 'Phase' in line or 'TODO' in line:
                continue
            # Keep copyright or license comments at the top
            if i < 10 and ('Copyright' in line or 'License' in line):
                cleaned_lines.append(line)
                continue
            continue
        
        # Remove inline comments but keep the code
        if '#' in line and not in_docstring:
            # Find the comment position (but not in strings)
            code_part = line
            in_string = False
            string_char = None
            comment_pos = -1
            
            for j, char in enumerate(line):
                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                elif char == '#' and not in_string:
                    comment_pos = j
                    break
            
            if comment_pos > 0:
                code_part = line[:comment_pos].rstrip() + '\n'
                # Only keep the line if there's actual code
                if code_part.strip():
                    cleaned_lines.append(code_part)
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    # Remove multiple consecutive empty lines
    final_lines = []
    prev_empty = False
    for line in cleaned_lines:
        if line.strip() == '':
            if not prev_empty:
                final_lines.append(line)
                prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False
    
    with open(output_file, 'w') as f:
        f.writelines(final_lines)
    
    return len(lines), len(final_lines)

if __name__ == "__main__":
    original, cleaned = clean_file('omega4_main.py', 'omega4_main_clean.py')
    print(f"Original lines: {original}")
    print(f"Cleaned lines: {cleaned}")
    print(f"Removed: {original - cleaned} lines")