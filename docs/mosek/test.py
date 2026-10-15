import mosek
from mosek.fusion import *

def main():
    # 1. Create a fresh optimization model context
    with Model("production_mix") as M:
        
        # 2. Define decision variables (x0 and x1 must be >= 0.0)
        x = M.variable("production_rate", 2, Domain.greaterThan(0.0))
        
        # 3. Create Constraints (e.g., resource allocations)
        # Constraint 1: 3*x0 + 1*x1 <= 9.0
        M.constraint("Resource_A", Expr.dot([3.0, 1.0], x), Domain.lessThan(9.0))
        
        # Constraint 2: 1*x0 + 2*x1 <= 8.0
        M.constraint("Resource_B", Expr.dot([1.0, 2.0], x), Domain.lessThan(8.0))
        
        # 4. Set Objective: Maximize profit (Profit = 2*x0 + 3*x1)
        M.objective("Total_Profit", ObjectiveSense.Maximize, Expr.dot([2.0, 3.0], x))
        
        # 5. Execute the solver engine
        print("--- Starting MOSEK Optimization Engine ---")
        M.solve()
        
        # 6. Extract and output the structural results
        print("\n🎉 Optimization Complete!")
        print(f"Status: {M.getPrimalSolutionStatus()}")
        print(f"Maximized Profit: {M.primalObjValue():.2f}")
        print(f"Optimal Value for Product 0 (x0): {x.level()[0]:.2f}")
        print(f"Optimal Value for Product 1 (x1): {x.level()[1]:.2f}")

if __name__ == "__main__":
    main()
