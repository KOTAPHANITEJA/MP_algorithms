import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from django.shortcuts import render
from django.http import HttpResponse
import io
import base64

def home(request):
    return render(request, 'home.html')


def parse_objective(objective_function):
    # Parse the objective function string and return coefficients
    # Example: "Z = 5x + 4y" -> [5, 4]
    try:
        parts = objective_function.split('=')[1].strip().split('+')
        c = [float(part.split('x')[0].strip()) for part in parts]
        return c
    except:
        return None

def parse_constraint(constraint):
    # Parse the constraint string and return coefficients and RHS
    # Example: "1x + 2y <= 20" -> ([1, 2], 20)
    try:
        lhs, rhs = constraint.split('<=')
        rhs = float(rhs.strip())
        parts = lhs.strip().split('+')
        coeff = [float(part.split('x')[0].strip()) for part in parts]
        return coeff, rhs
    except:
        return None

def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None):
    """Plots the constraints, feasible region, and optimal solution."""
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    # Plot constraints as lines
    for coeff, b in constraints:
        if coeff[1] != 0:  # Plot lines with a slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    if feasible_region is not None and len(feasible_region) > 0:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
        plt.gca().add_patch(polygon)

    # Highlight corner points
    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')  # Mark corners

    # Highlight the optimal solution
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    return string.decode('utf-8')

def solve_linear_program(c, A, b):
    """Solve the linear programming problem and plot."""
    bounds = [0, max(b)]  # Define a reasonable range for visualization
    constraints = list(zip(A, b))

    # Solve using vertices of the feasible region
    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            # Find intersection of two lines
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if all(np.dot(A, vertex) <= b) and all(vertex >= 0):  # Ensure non-negativity and feasibility
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    # Filter unique vertices
    feasible_vertices = np.unique(vertices, axis=0)

    # Evaluate the objective function at each vertex
    if len(feasible_vertices) > 0:
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]

        solution = f"Optimal Point: {optimal_vertex}, Optimal Value: {optimal_value}"
        graph = plot_constraints(constraints, bounds, feasible_region=feasible_vertices, optimal_vertex=optimal_vertex)
        return solution, graph
    else:
        return "No feasible region found.", None

def graphical_method(request):
    if request.method == "POST":
        objective_function = request.POST.get("objective_function")
        constraints_input = request.POST.getlist("constraints")

        # Parse the objective function
        c = parse_objective(objective_function)
        if not c:
            return render(request, 'graphical_method.html', {'error': 'Invalid objective function format'})

        # Parse each constraint
        parsed_constraints = []
        for constraint in constraints_input:
            parsed = parse_constraint(constraint)
            if parsed:
                parsed_constraints.append(parsed)
            else:
                return render(request, 'graphical_method.html', {'error': f'Invalid constraint format: {constraint}'})

        A = [parsed[0] for parsed in parsed_constraints]  # Collecting all A (coefficients)
        b = [parsed[1] for parsed in parsed_constraints]  # Collecting all b (right-hand sides)

        # Solve the linear program
        solution, graph = solve_linear_program(c, A, b)
        return render(request, 'graphical_method.html', {'solution': solution, 'graph': graph})
    else:
        return render(request, 'graphical_method.html')

def simplex_method(request):
    return render(request, 'simplex_method.html')