import pandas as pd
# TODO: Adjust for custom IndexDataset
def biased_quaternary_search(series, left, right, target):
    if right >= left:
        # Calculate the indices for the partition points
        part1 = left + (right - left) // 6  # less likely towards the start
        part2 = left + 2 * (right - left) // 4  # more likely around the middle
        part3 = left + 5 * (right - left) // 6  # less likely towards the end

        # Check the values at each partitioning point
        if series.iloc[part1] == target:
            return part1
        if series.iloc[part2] == target:
            return part2
        if series.iloc[part3] == target:
            return part3

        # Recursively search the appropriate segment
        if target < series.iloc[part1]:
            return biased_quaternary_search(series, left, part1 - 1, target)
        elif target < series.iloc[part2]:
            return biased_quaternary_search(series, part1 + 1, part2 - 1, target)
        elif target < series.iloc[part3]:
            return biased_quaternary_search(series, part2 + 1, part3 - 1, target)
        else:
            return biased_quaternary_search(series, part3 + 1, right, target)
    else:
        return -1