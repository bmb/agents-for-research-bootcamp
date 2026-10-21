#!/usr/bin/env python3
"""
Delivery Routing Problem - Baseline Solver using Python.

Two vans deliver boxes from one warehouse. Each van can carry at most 10 boxes.
Every customer must be visited exactly once, and every used van must start and finish at the warehouse.
Minimize total route distance. No time windows or service times.

Locations:
  Warehouse: (0, 0), 0 boxes
  A: (2, 1), 4 boxes
  B: (3, 2), 3 boxes
  C: (2, 4), 3 boxes
  D: (-2, 1), 4 boxes
  E: (-3, 2), 3 boxes
  F: (-2, 4), 3 boxes

Use Euclidean distances in arbitrary units.
"""

import math
from itertools import permutations


# Location data: (name, x, y, boxes)
LOCATIONS = [
    ("Warehouse", 0, 0, 0),
    ("A", 2, 1, 4),
    ("B", 3, 2, 3),
    ("C", 2, 4, 3),
    ("D", -2, 1, 4),
    ("E", -3, 2, 3),
    ("F", -2, 4, 3),
]

# Index constants: 0=Warehouse, 1=A, 2=B, 3=C, 4=D, 5=E, 6=F
WH_INDEX = 0
CUSTOMERS = [1, 2, 3, 4, 5, 6]  # A through F

VAN_CAPACITY = 10


def compute_euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_coords(loc_index):
    """Get (x, y) coordinates for a location index."""
    name, x, y, _ = LOCATIONS[loc_index]
    return x, y


def get_boxes(loc_index):
    """Get number of boxes at a location."""
    _, _, _, boxes = LOCATIONS[loc_index]
    return boxes


def compute_distance(i, j):
    """Euclidean distance between location i and j."""
    if i == j:
        return 0.0
    xi, yi = get_coords(i)
    xj, yj = get_coords(j)
    return compute_euclidean_distance(xi, yi, xj, yj)


def compute_route_distance(order):
    """Compute total distance for a route: Warehouse -> customers -> Warehouse.
    
    order: list of customer indices (1-6)
    """
    if not order:
        return 0.0
    
    total = compute_distance(WH_INDEX, order[0])  # Warehouse to first customer
    prev = order[0]
    for cust in order[1:]:
        total += compute_distance(prev, cust)
        prev = cust
    total += compute_distance(prev, WH_INDEX)  # Last customer to Warehouse
    return total


def find_optimal_routes():
    """Find optimal routes for two vans with capacity constraints.
    
    Returns:
        best_total_distance: minimum total distance found
        best_routes: dict with 'van1' and 'van2', each containing:
            - 'order': list of customer indices in visit order
            - 'distance': total route distance
    """
    
    n_customers = 6  # A through F
    total_boxes = sum(get_boxes(c) for c in CUSTOMERS)
    print(f"Total boxes to deliver: {total_boxes}")
    print(f"Van capacity: {VAN_CAPACITY}")
    print()
    
    # Best solution tracking
    best_total_distance = float('inf')
    best_routes = None
    
    # Try all possible ways to split customers into two groups
    # Each group must have total boxes <= van_capacity (10)
    
    # For each possible subset of customers assigned to van 1
    for mask in range(1 << n_customers):  # 2^6 = 64 possibilities
        van1_customers = []
        van2_customers = []
        
        for i in range(n_customers):
            if mask & (1 << i):
                van1_customers.append(CUSTOMERS[i])
            else:
                van2_customers.append(CUSTOMERS[i])
        
        # Check capacity constraints
        van1_boxes = sum(get_boxes(c) for c in van1_customers)
        van2_boxes = sum(get_boxes(c) for c in van2_customers)
        
        if van1_boxes > VAN_CAPACITY or van2_boxes > VAN_CAPACITY:
            continue
        
        # For each van, try all permutations of visit order
        # Compute best route distance and order for van 1
        van1_best_dist = float('inf')
        van1_best_order = None
        for perm in permutations(van1_customers):
            d = compute_route_distance(list(perm))
            if d < van1_best_dist:
                van1_best_dist = d
                van1_best_order = list(perm)
        
        # Compute best route distance and order for van 2
        van2_best_dist = float('inf')
        van2_best_order = None
        for perm in permutations(van2_customers):
            d = compute_route_distance(list(perm))
            if d < van2_best_dist:
                van2_best_dist = d
                van2_best_order = list(perm)
        
        total_dist = van1_best_dist + van2_best_dist
        
        if total_dist < best_total_distance:
            best_total_distance = total_dist
            best_routes = {
                'van1': {'order': van1_best_order, 'distance': van1_best_dist},
                'van2': {'order': van2_best_order, 'distance': van2_best_dist},
            }
    
    return best_total_distance, best_routes


def main():
    """Main function to solve the delivery routing problem."""
    
    best_total_distance, best_routes = find_optimal_routes()
    
    if best_routes is None:
        print("No feasible solution found!")
        return
    
    print("=" * 60)
    print("OPTIMAL SOLUTION FOUND")
    print("=" * 60)
    print(f"Total minimum distance: {best_total_distance:.2f}")
    print()
    
    # Calculate total boxes delivered (all customers should be visited)
    all_customer_indices = set()
    for van_id, van_info in best_routes.items():
        for c in van_info['order']:
            all_customer_indices.add(c)
    
    total_boxes_delivered = sum(get_boxes(c) for c in all_customer_indices)
    print(f"Total boxes delivered: {total_boxes_delivered}")
    print()
    
    # Print routes
    van_names = {'van1': 'Van 1', 'van2': 'Van 2'}
    cust_names = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
    
    for van_id in ['van1', 'van2']:
        van_name = van_names[van_id]
        cust_order = best_routes[van_id]['order']
        
        print(f"{van_name} route:")
        print(f"  Route: Warehouse -> ", end="")
        
        prev = WH_INDEX  # warehouse
        van_boxes = 0
        
        for i, cust in enumerate(cust_order):
            cust_boxes = get_boxes(cust)
            van_boxes += cust_boxes
            cust_name = cust_names[cust]
            if i > 0:
                print(f" -> {cust_name}", end="")
        
        # Print return to warehouse
        print(f" -> Warehouse")
        print(f"  Total boxes on this van: {van_boxes}")
        print(f"  Route distance: {compute_route_distance(cust_order):.2f}")
        print()
    
    # Compute precise distances for each van
    van1_order = best_routes['van1']['order']
    van2_order = best_routes['van2']['order']
    
    van1_dist = compute_route_distance(van1_order)
    van2_dist = compute_route_distance(van2_order)
    
    print("Detailed route distances:")
    print(f"Van 1 distance: {van1_dist:.2f}")
    print(f"Van 2 distance: {van2_dist:.2f}")
    print(f"Total distance: {van1_dist + van2_dist:.2f}")
    print()
    
    # Print each van's detailed route path
    print("Van 1 route path:")
    prev = WH_INDEX
    for cust in van1_order:
        cust_name = cust_names[cust]
        d = compute_distance(prev, cust)
        print(f"  {cust_name} (idx {cust}) from warehouse: distance {d:.2f}")
        prev = cust
    d_return = compute_distance(prev, WH_INDEX)
    print(f"  Return to warehouse: distance {d_return:.2f}")
    print(f"  Van 1 total: {compute_route_distance(van1_order):.2f}")
    print()
    
    print("Van 2 route path:")
    prev = WH_INDEX
    for cust in van2_order:
        cust_name = cust_names[cust]
        d = compute_distance(prev, cust)
        print(f"  {cust_name} (idx {cust}) from warehouse: distance {d:.2f}")
        prev = cust
    d_return = compute_distance(prev, WH_INDEX)
    print(f"  Return to warehouse: distance {d_return:.2f}")
    print(f"  Van 2 total: {compute_route_distance(van2_order):.2f}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Compute customers visited for each van
    van1_customers = best_routes['van1']['order']
    van2_customers = best_routes['van2']['order']
    
    van1_cust_names = [ {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}[c] for c in van1_customers ]
    van2_cust_names = [ {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}[c] for c in van2_customers ]
    
    van1_boxes = sum(get_boxes(c) for c in van1_customers)
    van2_boxes = sum(get_boxes(c) for c in van2_customers)
    
    print(f"Van 1: visits customers {van1_cust_names}")
    print(f"  Boxes: {van1_boxes}")
    print(f"  Route distance: {compute_route_distance(van1_customers):.2f}")
    print()
    print(f"Van 2: visits customers {van2_cust_names}")
    print(f"  Boxes: {van2_boxes}")
    print(f"  Route distance: {compute_route_distance(van2_customers):.2f}")
    print()
    print(f"Total distance: {best_total_distance:.2f}")
    print(f"Total boxes delivered: {total_boxes_delivered}/{sum(get_boxes(c) for c in CUSTOMERS)}")
    print()
    print("Route details:")
    print(f"  Van 1 carries {van1_boxes} boxes (capacity: {VAN_CAPACITY})")
    print(f"  Van 2 carries {van2_boxes} boxes (capacity: {VAN_CAPACITY})")
    
    # Save results to results folder
    import json
    import os
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results summary
    results = {
        "problem": "Delivery Routing - Two vans, capacity 10 boxes each",
        "locations": {
            "Warehouse": {"x": 0, "y": 0, "boxes": 0},
            "A": {"x": 2, "y": 1, "boxes": 4},
            "B": {"x": 3, "y": 2, "boxes": 3},
            "C": {"x": 2, "y": 4, "boxes": 3},
            "D": {"x": -2, "y": 1, "boxes": 4},
            "E": {"x": -3, "y": 2, "boxes": 3},
            "F": {"x": -2, "y": 4, "boxes": 3},
        },
        "van_capacity": 10,
        "total_boxes": 20,
        "solution": {
            "total_minimum_distance": best_total_distance,
            "van1": {
                "customer_names": van1_cust_names,
                "boxes": van1_boxes,
                "route_distance": van1_dist
            },
            "van2": {
                "customer_names": van2_cust_names,
                "boxes": van2_boxes,
                "route_distance": van2_dist
            }
        },
        "route_distances": {
            "van1": van1_dist,
            "van2": van2_dist,
            "total": best_total_distance
        }
    }
    
    # Write results to JSON file
    results_path = os.path.join(results_dir, "delivery_routing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()