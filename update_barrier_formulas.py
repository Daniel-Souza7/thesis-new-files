#!/usr/bin/env python3
"""
Script to update barrier option formulas in payoffs_index.tex to use normalized notation.
"""

import re

def update_max_min_barriers(content):
    """Update MaxCall/MinPut barrier variants."""
    count = 0

    # Pattern for MaxCall: Replace max_i S_i(t) in payoff with normalized version
    # Match: \max(0, \max_i S_i(t) - K) or \max\left(0, \max_i S_i(t) - K\right)
    pattern1 = r'(\\max\\left\(0,\s*)\\max_i S_i\(t\)(\s*-\s*K\\right\))'
    replacement1 = r'\1\\max_i \\frac{S_i(t)}{S_i(0)}\2'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1

    pattern1b = r'(\\max\(0,\s*)\\max_i S_i\(t\)(\s*-\s*K\))'
    replacement1b = r'\\max\\left(0, \\max_i \\frac{S_i(t)}{S_i(0)}\2'
    content, n1b = re.subn(pattern1b, replacement1b, content)
    count += n1b

    # Pattern for MinPut: Replace min_i S_i(t) in payoff with normalized version
    pattern2 = r'(\\max\\left\(0,\s*K\s*-\s*)\\min_i S_i\(t\)(\\right\))'
    replacement2 = r'\1\\min_i \\frac{S_i(t)}{S_i(0)}\2'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2

    pattern2b = r'(\\max\(0,\s*K\s*-\s*)\\min_i S_i\(t\)(\))'
    replacement2b = r'\\max\\left(0, K - \\min_i \\frac{S_i(t)}{S_i(0)}\\right)'
    content, n2b = re.subn(pattern2b, replacement2b, content)
    count += n2b

    print(f"  MaxCall: {n1 + n1b}, MinPut: {n2 + n2b}")
    return content, count

def update_range_basket_barriers(content):
    """Update MaxDispersion basket barrier variants."""
    count = 0

    # Pattern for MaxDispersionCall: max(0, [max_i S_i(t) - min_i S_i(t)] - K)
    pattern1 = r'(\\left\[)\\max_i S_i\(t\)\s*-\s*\\min_i S_i\(t\)(\\right\]\s*-\s*K)'
    replacement1 = r'\1\\max_i \\frac{S_i(t)}{S_i(0)} - \\min_i \\frac{S_i(t)}{S_i(0)}\2'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1

    # Pattern for MaxDispersionPut: max(0, K - [max_i S_i(t) - min_i S_i(t)])
    pattern2 = r'(K\s*-\s*\\left\[)\\max_i S_i\(t\)\s*-\s*\\min_i S_i\(t\)(\\right\])'
    replacement2 = r'\1\\max_i \\frac{S_i(t)}{S_i(0)} - \\min_i \\frac{S_i(t)}{S_i(0)}\2'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2

    print(f"  MaxDispersionCall: {n1}, MaxDispersionPut: {n2}")
    return content, count

def update_dispersion_barriers(content):
    """Update Dispersion barrier variants - these have WRONG formulas that need complete replacement."""
    count = 0

    # DispersionCall barriers - replace the WRONG formula with correct normalized formula
    # Current (WRONG): \max\left(0, \sum_{i=1}^d S_i(t) - \overline{S}(t) \right)
    # Correct: \max\left(0, \sigma_{\text{norm}}(t) - K\right)

    old_call_pattern = r'\\max\\left\(0,\s*\\sum_\{i=1\}\^d S_i\(t\)\s*-\s*\\overline\{S\}\(t\)\s*\\right\)'
    new_call_formula = r'\\max\\left(0, \\sigma_{\\text{norm}}(t) - K\\right)'

    # Need to also update the "where" clause
    old_where_call = r' where \$\\overline\{S\}\(t\) = \\frac\{1\}\{d\}\\sum_\{i=1\}\^d S_i\(t\)\$'
    new_where = r' where $\\sigma_{\\text{norm}}(t) = \\sqrt{\\frac{1}{d}\\sum_{i=1}^d \\left(\\frac{S_i(t)}{S_i(0)} - \\overline{R}(t)\\right)^2}$ and $\\overline{R}(t) = \\frac{1}{d}\\sum_{i=1}^d \\frac{S_i(t)}{S_i(0)}$'

    # First replace the formulas
    content, n1 = re.subn(old_call_pattern, new_call_formula, content)
    count += n1

    # DispersionPut barriers
    # Current (WRONG): \max\left(0, - \sum_{i=1}^d S_i(t) - \overline{S}(t)\right)
    # Note the weird "-" sign at the beginning
    # Correct: \max\left(0, K - \sigma_{\text{norm}}(t)\right)

    old_put_pattern = r'\\max\\left\(0,\s*-\s*\\sum_\{i=1\}\^d S_i\(t\)\s*-\s*\\overline\{S\}\(t\)\\right\)'
    new_put_formula = r'\\max\\left(0, K - \\sigma_{\\text{norm}}(t)\\right)'
    content, n2 = re.subn(old_put_pattern, new_put_formula, content)
    count += n2

    # Now replace all the "where" clauses for dispersion
    content, n3 = re.subn(old_where_call, new_where, content)

    print(f"  DispersionCall: {n1}, DispersionPut: {n2}, Where clauses: {n3}")
    return content, count

def update_bestofk_worstofk_barriers(content):
    """Update BestOfK/WorstOfK barrier variants."""
    count = 0

    # Pattern for BestOfKCall: \sum_{i=1}^k S_{(i)}(t) -> \sum_{i=1}^k R_{(i)}(t)
    pattern1 = r'(\\sum_\{i=1\}\^\{?k\}?\s*)S_\{\(i\)\}\(t\)'
    replacement1 = r'\1R_{(i)}(t)'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1

    # Pattern for WorstOfKPut: \sum_{i=d-k+1}^d S_{(i)}(t) -> \sum_{i=d-k+1}^d R_{(i)}(t)
    pattern2 = r'(\\sum_\{i=d-k\+1\}\^\{?d\}?\s*)S_\{\(i\)\}\(t\)'
    replacement2 = r'\1R_{(i)}(t)'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2

    # Update the where clause explanation for BestOfK
    old_where_bestofk = r' where \$S_\{\(1\)\} \\geq S_\{\(2\)\} \\geq \\ldots \\geq S_\{\(d\)\}\$'
    new_where_bestofk = r' where $R_{(1)} \\geq R_{(2)} \\geq \\ldots \\geq R_{(d)}$ are sorted returns $R_i(t) = \\frac{S_i(t)}{S_i(0)}$'
    content, n3 = re.subn(old_where_bestofk, new_where_bestofk, content)

    # Shorter version for WorstOfK
    old_where_worstofk = r' where \$S_\{\(1\)\} \\geq \\ldots \\geq S_\{\(d\)\}\$'
    new_where_worstofk = r' where $R_{(1)} \\geq \\ldots \\geq R_{(d)}$ are sorted returns $R_i(t) = \\frac{S_i(t)}{S_i(0)}$'
    content, n4 = re.subn(old_where_worstofk, new_where_worstofk, content)

    print(f"  BestOfK sums: {n1}, WorstOfK sums: {n2}, Where clauses: {n3 + n4}")
    return content, count

def update_rankweighted_barriers(content):
    """Update RankWeightedBasket barrier variants."""
    count = 0

    # Pattern: \sum_{i=1}^d w_i S_{(i)}(t) -> \sum_{i=1}^d w_i R_{(i)}(t)
    pattern = r'(\\sum_\{i=1\}\^\{?d\}?\s*w_i\s*)S_\{\(i\)\}\(t\)'
    replacement = r'\1R_{(i)}(t)'
    content, n = re.subn(pattern, replacement, content)
    count += n

    # Update the where clause - need to change S_{(1)} >= ... >= S_{(d)} to R_{(1)} >= ... >= R_{(d)}
    # and add explanation
    old_where = r'(where \$w_i = \\frac\{d\+1-i\}\{\\sum_\{j=1\}\^d j\}\$) and \$S_\{\(1\)\} \\geq \\ldots \\geq S_\{\(d\)\}\$'
    new_where = r'\1 and $R_{(1)} \\geq \\ldots \\geq R_{(d)}$ are sorted returns $R_i(t) = \\frac{S_i(t)}{S_i(0)}$'
    content, n2 = re.subn(old_where, new_where, content)

    print(f"  RankWeighted sums: {n}, Where clauses: {n2}")
    return content, count

def main():
    # Read the file
    with open('/home/user/thesis-new-files/payoffs_index.tex', 'r') as f:
        content = f.read()

    original_content = content
    total_count = 0

    # Update each category
    print("Updating MaxCall/MinPut barriers...")
    content, count = update_max_min_barriers(content)
    total_count += count

    print("Updating MaxDispersion basket barriers...")
    content, count = update_range_basket_barriers(content)
    total_count += count

    print("Updating Dispersion barriers...")
    content, count = update_dispersion_barriers(content)
    total_count += count

    print("Updating BestOfK/WorstOfK barriers...")
    content, count = update_bestofk_worstofk_barriers(content)
    total_count += count

    print("Updating RankWeightedBasket barriers...")
    content, count = update_rankweighted_barriers(content)
    total_count += count

    # Write the file back
    if content != original_content:
        with open('/home/user/thesis-new-files/payoffs_index.tex', 'w') as f:
            f.write(content)
        print(f"\nTotal replacements: {total_count}")
        print("File updated successfully!")
    else:
        print("\nNo changes made to file")

    return total_count

if __name__ == '__main__':
    count = main()
