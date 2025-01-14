from data_module import data
from vrpstates import CvrptwState

def verify_time_windows(data: dict, sol: CvrptwState, percentage: bool = False) -> dict:
    """
    Verifies the time windows of the solution and returns the early/late/ontime counts.
    If percentage is True, then the counts are returned as percentages of served customers.
        Parameters:
            data: dict
                The data dictionary.
            sol: CvrptwState
                The solution to be verified.
            percentage: bool
                If True, the counts are returned as percentages.
    """
    late = 0
    early = 0
    ontime = 0
    sum_early = 0
    sum_late = 0

    for route in sol.routes:
        for cust_idx, customer in enumerate(route.customers_list):
            if route.planned_windows[cust_idx][0] < data["time_window"][customer][0]:
                early += 1
                sum_early += data["time_window"][customer][0] - route.planned_windows[cust_idx][0]
            elif route.planned_windows[cust_idx][0] > data["time_window"][customer][1]:
                late += 1
                sum_late += route.planned_windows[cust_idx][1] - data["time_window"][customer][0]
            else:
                ontime += 1
    total = early + late + ontime
    
    if percentage:
        early = round(early / total * 100, 1)
        late = round(late / total * 100, 1)
        ontime = round(ontime / total * 100, 1)

    stats = {
        "total_served": total,
        "early": early,
        "late": late,
        "ontime": ontime,
        "sum_early": sum_early,
        "sum_late": sum_late
    }
    return stats
