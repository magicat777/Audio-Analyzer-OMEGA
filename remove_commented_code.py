#!/usr/bin/env python3
"""Remove all commented code blocks and lines from the file"""

def remove_commented_code(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Keep shebang
        if i == 0 and line.startswith('#!'):
            cleaned_lines.append(line)
            i += 1
            continue
        
        # Skip comment blocks (lines that are just comments)
        if line.strip().startswith('#'):
            i += 1
            continue
            
        # Handle inline comments
        if '#' in line:
            # Check if # is inside a string
            in_string = False
            string_char = None
            cleaned_line = ""
            
            for j, char in enumerate(line):
                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                
                if char == '#' and not in_string:
                    # Found a comment, stop here
                    break
                    
                cleaned_line += char
            
            # Only add if there's actual code
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line.rstrip())
        else:
            # No comments in this line
            cleaned_lines.append(line)
            
        i += 1
    
    # Join and clean up excessive blank lines
    result = '\n'.join(cleaned_lines)
    
    # Replace multiple blank lines with just two
    while '\n\n\n\n' in result:
        result = result.replace('\n\n\n\n', '\n\n\n')
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    with open(output_file, 'w') as f:
        f.write(result)
    
    return len(lines), len(result.split('\n'))

if __name__ == "__main__":
    # Restore original first
    import shutil
    shutil.copy('omega4_main_with_comments.py', 'omega4_main.py')
    
    original, cleaned = remove_commented_code('omega4_main.py', 'omega4_main_clean.py')
    print(f"Original lines: {original}")
    print(f"Cleaned lines: {cleaned}")
    print(f"Removed: {original - cleaned} lines")