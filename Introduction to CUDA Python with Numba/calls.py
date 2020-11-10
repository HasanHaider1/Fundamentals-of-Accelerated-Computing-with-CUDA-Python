# Use the 'File' menu above to 'Save' after pasting in your 3 function calls.
%%timeit
# Feel free to modify the 3 function calls in this cell
normalize(greyscales, out = normalized)
weigh(normalized, weights, out = weighted)
SOLUTION = activate(weighted)