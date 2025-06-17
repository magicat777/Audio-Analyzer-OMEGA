#!/usr/bin/env python3
"""Remove ALL commented lines from omega4_main.py"""

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
        
        # Skip ALL comment lines
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        
        # For code lines with inline comments, keep only the code part
        if '#' in line and not in_docstring:
            # Find the comment position (but not in strings)
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
                code_part = line[:comment_pos].rstrip()
                # Only keep the line if there's actual code
                if code_part.strip():
                    cleaned_lines.append(code_part + '\n')
            else:
                cleaned_lines.append(line)
        else:
            # Keep lines without comments
            if line.strip() or line == '\n':  # Keep non-empty lines and single newlines
                cleaned_lines.append(line)
    
    # Remove excessive empty lines (more than 2 consecutive)
    final_lines = []
    empty_count = 0
    
    for line in cleaned_lines:
        if line.strip() == '':
            empty_count += 1
            if empty_count <= 2:
                final_lines.append(line)
        else:
            empty_count = 0
            final_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(final_lines)
    
    return len(lines), len(final_lines)

if __name__ == "__main__":
    original, cleaned = clean_file('omega4_main.py', 'omega4_main_clean.py')
    print(f"Original lines: {original}")
    print(f"Cleaned lines: {cleaned}")
    print(f"Removed: {original - cleaned} lines")