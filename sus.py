arrival = [1, 3, 3, 3, 5, 7]
duration = [2, 2, 1, 1, 2, 1]


def maxEvents():
    events = [0]
    arr, dur = zip(*sorted(zip(arrival, duration)))  # Splits into pairs of arr/dur,
    # then puts back into tuples sorted but still paired
    print(arr, dur)

    for i in range(len(arrival) - 1):
        print()
        print(f'Events before: {events}')
        index = events[-1]  # Stays same for multiple arrivals at same time, then skips to next arrival
        next_arr, next_dur = arr[i + 1], dur[i + 1]
        # print(f'arr[{index}]: {arr[index]} | dur[{index}]: {dur[index]}')
        # print(f'next_arr: {next_arr} | next_dur: {next_dur}')

        print(f'Checking if {arr[index]} + {dur[index]} <= {next_arr}...')
        # Check if arr + dur <= next arrival time (can fit in slot); if yes, add to counter
        if arr[index] + dur[index] <= next_arr:
            print(f'{arr[index]} + {dur[index]} <= {next_arr}, Appending {i + 1}')
            events.append(i + 1)
        # Removes longer events to make room for shorter (or equal) one in same slot
        elif arr[index] + dur[index] >= next_arr + next_dur:
            print(f'{arr[index]} + {dur[index]} >= {next_arr} + {next_dur}, Popping {events[-1]}, adding {i + 1}')
            events.pop(-1)
            events.append(i + 1)
        print(f'Events after: {events}')

    res = []
    for e in events:
        res.append((arr[e], dur[e]))
    print(res)
    return len(events)


print(maxEvents())
