import os
import re
import glob

def fix_unpacking_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    modified = False

    new_lines = []
    for line in lines:
        if 'element.project(' in line and '=' in line and '==' not in line:
            parts = line.split('=', 1)
            lhs = parts[0]

            # Count commas in lhs to see how many variables are being unpacked
            if lhs.count(',') == 1:
                # E.g., `V_n[n], c_n[n] = ...` -> `V_n[n], c_n[n], _ = ...`
                if not lhs.strip().endswith(', _'):
                     new_line = line.replace(lhs, lhs.rstrip() + ', _', 1)
                     line = new_line
                     modified = True
            elif lhs.count(',') == 2:
                # Already unpacking 3, maybe? Like `V_n[n], c_n[n], _ =`
                pass

        new_lines.append(line)

    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Fixed {filepath}")

for ext in ['py']:
    for filepath in glob.glob(f'**/*.{ext}', recursive=True):
        fix_unpacking_in_file(filepath)
