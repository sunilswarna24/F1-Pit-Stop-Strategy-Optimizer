import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class SilverstoneSimulator:
    """
    Main class for simulating F1 race strategies at Silverstone GP.
    Implements Dynamic Programming and Greedy algorithms to optimize pit stop strategies.
    """
    
    def __init__(self):
        """
        Initialize the simulator with realistic F1 race parameters for Silverstone GP.
        All parameters are based on real-world F1 data and research.
        """
        self.total_laps = 52  # Total race distance for Silverstone GP
        
        # Tire compound characteristics - each has different performance vs durability trade-offs
        self.tire_compounds = {
            'soft': {
                'base_time': 89.0,      # Fastest lap time (seconds) when fresh
                'deg_rate': 0.030,      # How quickly the tire degrades (higher = faster wear)
                'fuel_cons': 2.2        # Fuel consumption per lap (kg) - soft tires need more energy
            },
            'medium': {
                'base_time': 91.0,      # Medium performance when fresh
                'deg_rate': 0.020,      # Moderate degradation rate
                'fuel_cons': 2.1        # Standard fuel consumption
            },
            'hard': {
                'base_time': 93.0,      # Slowest when fresh but most durable
                'deg_rate': 0.015,      # Slowest degradation - lasts longest
                'fuel_cons': 2.0        # Most fuel efficient
            }
        }
        
        # Race simulation parameters
        self.pit_loss = 21.0            # Time penalty for pit stop (seconds) - realistic Silverstone value
        self.fuel_capacity = 110.0      # Maximum fuel tank capacity (kg)
        self.fuel_effect = 0.25         # How much lap time increases per kg of fuel (seconds/kg)
        self.wear_threshold = 0.80      # When tire wear reaches 80%, performance drops significantly

    def tire_degradation(self, compound, tire_age, fuel_load):
        """
        Calculate current tire wear based on compound type, age, and fuel load.
        
        Args:
            compound: Tire type ('soft', 'medium', 'hard')
            tire_age: Number of laps on current tires
            fuel_load: Current fuel weight (kg)
            
        Returns:
            Wear level (0.0 = fresh, 1.0 = completely worn)
        """
        params = self.tire_compounds[compound]
        
        # Base degradation increases linearly with tire age
        base_deg = params['deg_rate'] * tire_age
        
        # Heavy fuel loads cause additional tire stress (5% impact max)
        fuel_impact = 0.05 * (fuel_load / self.fuel_capacity)
        
        # Total wear cannot exceed 95% in simulation
        return min(base_deg * (1 + fuel_impact), 0.95)

    def lap_time(self, compound, tire_age, fuel_load, pit_stop=False):
        """
        Calculate lap time based on current tire condition, fuel load, and pit stop penalty.
        
        Args:
            compound: Current tire compound
            tire_age: Laps completed on current tires
            fuel_load: Current fuel weight
            pit_stop: Whether this lap includes a pit stop
            
        Returns:
            Lap time in seconds (or infinity if tires are too worn)
        """
        # Prevent division by zero and negative values
        if tire_age == 0:
            tire_age = 1
        if fuel_load <= 0:
            fuel_load = 0.1
            
        # Calculate current tire wear
        wear = self.tire_degradation(compound, tire_age, fuel_load)
        
        # If tires are too worn, force a pit stop (infinite time penalty)
        if wear >= self.wear_threshold:
            return float('inf')
        
        params = self.tire_compounds[compound]
        
        # Progressive time penalty based on tire wear (non-linear performance drop)
        if wear < 0.3:
            # Fresh tires: minimal penalty
            time_penalty = wear * 1.0
        elif wear < 0.6:
            # Medium wear: moderate penalty increase
            time_penalty = 0.3 + (wear - 0.3) * 3.0
        else:
            # High wear: steep performance cliff
            time_penalty = 1.2 + (wear - 0.6) * 8.0
        
        # Fuel weight slows the car (heavier = slower)
        fuel_penalty = self.fuel_effect * fuel_load
        
        # Add pit stop time penalty if stopping this lap
        pit_penalty = self.pit_loss if pit_stop else 0
        
        # Calculate total lap time
        total_time = params['base_time'] + time_penalty + fuel_penalty + pit_penalty
        
        # Cap maximum reasonable lap time at 200 seconds
        return total_time if total_time < 200 else float('inf')

    def dynamic_programming_strategy(self):
        """
        Implement Dynamic Programming algorithm to find optimal pit stop strategy.
        Uses Bellman equations to work backwards from race end to find globally optimal decisions.
        
        Returns:
            Dictionary containing optimal actions for each race state
        """
        # DP table: dp[lap][(compound, tire_age)] = (total_time, action)
        dp = defaultdict(lambda: defaultdict(lambda: (float('inf'), None)))
        
        # Base case: Initialize final lap (lap 52) - race is finished, no more time needed
        for compound in self.tire_compounds:
            for age in range(35):  # Maximum realistic tire age
                dp[self.total_laps][(compound, age)] = (0, 'finish')

        # Work backwards from race end to beginning (Dynamic Programming principle)
        for lap in range(self.total_laps-1, -1, -1):
            for compound in self.tire_compounds:
                for age in range(1, 35):  # Start from age 1 (fresh tires)
                    
                    # Calculate remaining fuel at this point in race
                    fuel_used = lap * 2.1  # Average fuel consumption estimate
                    fuel = max(self.fuel_capacity - fuel_used, 0)
                    
                    # OPTION 1: Continue on current tires
                    next_age = age + 1
                    if next_age < 35:
                        # Calculate time for this lap + optimal time for remaining race
                        time_cont = self.lap_time(compound, next_age, fuel)
                        if time_cont < float('inf'):
                            next_time, _ = dp[lap+1][(compound, next_age)]
                            if next_time < float('inf'):
                                total_cont = time_cont + next_time
                            else:
                                total_cont = float('inf')
                        else:
                            total_cont = float('inf')
                    else:
                        total_cont = float('inf')
                    
                    # OPTION 2: Pit stop and change to different compound
                    best_pit_time = float('inf')
                    best_new_comp = compound
                    
                    # Try all other compounds as pit stop options
                    for new_comp in self.tire_compounds:
                        if new_comp == compound:
                            continue  # Can't change to same compound
                            
                        # Calculate pit stop lap time + optimal remaining race time
                        pit_time = self.lap_time(compound, age, fuel, pit_stop=True)
                        if pit_time < float('inf'):
                            next_time_pit, _ = dp[lap+1][(new_comp, 1)]  # Fresh tires after pit
                            if next_time_pit < float('inf'):
                                total_pit = pit_time + next_time_pit
                                if total_pit < best_pit_time:
                                    best_pit_time = total_pit
                                    best_new_comp = new_comp
                    
                    # Choose optimal action: continue vs pit stop
                    if total_cont <= best_pit_time and total_cont < float('inf'):
                        dp[lap][(compound, age)] = (total_cont, 'continue')
                    elif best_pit_time < float('inf'):
                        dp[lap][(compound, age)] = (best_pit_time, ('pit', best_new_comp))
        
        return dp

    def simulate_dp_race(self, start_compound='medium'):
        """
        Execute a complete race using the optimal Dynamic Programming strategy.
        
        Args:
            start_compound: Which tire compound to start the race with
            
        Returns:
            Tuple containing (lap_times, pit_laps, wear_history, fuel_history, pit_strategies)
        """
        # Get optimal strategy from DP algorithm
        dp = self.dynamic_programming_strategy()
        
        # Initialize race state
        current_comp = start_compound        # Current tire compound
        tire_age = 1                        # Laps on current tires
        fuel = self.fuel_capacity           # Starting fuel load
        
        # Track race progress
        lap_times = []          # Time for each lap
        pit_laps = []           # Which laps had pit stops
        pit_strategies = []     # Record of compound changes
        wear_history = []       # Tire wear progression
        fuel_history = [fuel]   # Fuel level progression
        
        # Simulate each lap of the race
        for lap in range(self.total_laps):
            # Look up optimal action from DP table
            if (current_comp, tire_age) in dp[lap]:
                time, action = dp[lap][(current_comp, tire_age)]
            else:
                action = 'continue'  # Fallback if state not found
            
            # Execute the optimal action
            if action == 'continue':
                # Stay on current tires
                lt = self.lap_time(current_comp, tire_age, fuel)
                if lt == float('inf'):
                    lt = 95.0  # Fallback lap time
                lap_times.append(lt)
                tire_age += 1  # Tires get one lap older
            else:
                # Pit stop: change to new compound
                new_comp = action[1]
                lt = self.lap_time(current_comp, tire_age, fuel, pit_stop=True)
                if lt == float('inf'):
                    lt = 115.0  # Fallback pit lap time
                lap_times.append(lt)
                pit_laps.append(lap+1)  # Record pit stop lap
                pit_strategies.append(f"Lap {lap+1}: {current_comp.upper()} → {new_comp.upper()}")
                current_comp = new_comp  # Switch to new compound
                tire_age = 1  # Fresh tires
            
            # Update fuel consumption (compound-specific)
            fuel_consumed = self.tire_compounds[current_comp]['fuel_cons']
            fuel -= fuel_consumed
            fuel_history.append(max(fuel, 0))  # Fuel can't go negative
            
            # Track tire wear
            wear = self.tire_degradation(current_comp, tire_age, fuel)
            wear_history.append(wear)
        
        return lap_times, pit_laps, wear_history, fuel_history, pit_strategies

    def simulate_greedy_race(self, start_compound='medium'):
        """
        Execute a race using a simple greedy strategy (local optimization).
        Makes pit stop decisions based on current conditions only, not future optimization.
        
        Args:
            start_compound: Starting tire compound
            
        Returns:
            Same format as simulate_dp_race()
        """
        # Initialize race state
        current_comp = start_compound
        tire_age = 1
        fuel = self.fuel_capacity
        
        # Track race progress
        lap_times = []
        pit_laps = []
        pit_strategies = []
        wear_history = []
        fuel_history = [fuel]
        last_pit_lap = -20  # Track when we last pitted (start with large negative)
        
        # Simulate each lap
        for lap in range(self.total_laps):
            wear = self.tire_degradation(current_comp, tire_age, fuel)
            
            # Greedy decision logic: pit only when absolutely necessary
            should_pit = False
            
            # Only consider pitting if enough time has passed since last pit
            if (lap - last_pit_lap >= 20 and     # Minimum 20 laps between pits
                lap < self.total_laps - 10):     # Don't pit in final 10 laps
                
                # Critical conditions that force a pit stop
                extreme_wear = wear >= 0.75      # Very high tire wear
                extreme_age = tire_age >= 30     # Very old tires
                fuel_critical = fuel < 5         # Almost out of fuel
                
                # Pit if any critical condition is met
                if extreme_wear or extreme_age or fuel_critical:
                    should_pit = True
            
            # Execute pit stop decision
            if should_pit:
                # Choose new compound based on remaining race distance
                remaining_laps = self.total_laps - lap
                candidates = [c for c in self.tire_compounds if c != current_comp]
                
                # Strategic compound selection
                if remaining_laps > 35:
                    # Long stint ahead - choose hard (most durable)
                    best_comp = 'hard' if 'hard' in candidates else candidates[0]
                elif remaining_laps > 20:
                    # Medium stint - choose medium
                    best_comp = 'medium' if 'medium' in candidates else candidates[0]
                else:
                    # Sprint to finish - choose soft (fastest)
                    best_comp = 'soft' if 'soft' in candidates else candidates[0]
                
                # Execute pit stop
                lt = self.lap_time(current_comp, tire_age, fuel, pit_stop=True)
                if lt == float('inf'):
                    lt = 115.0
                lap_times.append(lt)
                pit_laps.append(lap+1)
                pit_strategies.append(f"Lap {lap+1}: {current_comp.upper()} → {best_comp.upper()}")
                current_comp = best_comp
                tire_age = 1
                last_pit_lap = lap
            else:
                # Continue on current tires
                lt = self.lap_time(current_comp, tire_age, fuel)
                if lt == float('inf'):
                    # Emergency handling for extremely worn tires
                    wear_capped = min(wear, 0.85)
                    lt = (self.tire_compounds[current_comp]['base_time'] + 
                          wear_capped * 6.0 + 
                          self.fuel_effect * fuel)
                lap_times.append(lt)
                tire_age += 1
            
            # Update fuel and wear tracking
            fuel_consumed = self.tire_compounds[current_comp]['fuel_cons']
            fuel -= fuel_consumed
            fuel_history.append(max(fuel, 0))
            final_wear = self.tire_degradation(current_comp, tire_age, fuel)
            wear_history.append(final_wear)
        
        return lap_times, pit_laps, wear_history, fuel_history, pit_strategies

    def find_optimal_starting_compound(self):
        """
        Test all three tire compounds as starting options to find the optimal choice.
        
        Returns:
            Tuple of (all_results_dict, best_dp_compound, best_greedy_compound)
        """
        results = {}
        
        # Test each compound as starting tire
        for compound in self.tire_compounds:
            dp_result = self.simulate_dp_race(compound)
            greedy_result = self.simulate_greedy_race(compound)
            
            # Store results and calculate total race times
            results[compound] = {
                'dp': dp_result,
                'greedy': greedy_result,
                'dp_time': sum(dp_result[0]),      # Total DP race time
                'greedy_time': sum(greedy_result[0])  # Total greedy race time
            }
        
        # Find best starting compound for each strategy
        best_dp = min(results.keys(), key=lambda x: results[x]['dp_time'])
        best_greedy = min(results.keys(), key=lambda x: results[x]['greedy_time'])
        
        return results, best_dp, best_greedy

    def plot_comparison(self, dp_results, greedy_results, dp_strategies, greedy_strategies):
        """
        Create main comparison plot showing DP vs Greedy strategy performance.
        
        Args:
            dp_results: Results from DP strategy simulation
            greedy_results: Results from greedy strategy simulation
            dp_strategies: List of DP compound changes
            greedy_strategies: List of greedy compound changes
        """
        # Create 2x2 subplot layout with enhanced size for visibility
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("Silverstone GP: DP vs Greedy Strategy (Enhanced Visibility)", fontsize=18)
        
        # SUBPLOT 1: Lap time progression with pit stop annotations
        axes[0,0].plot(dp_results[0], 'b-', linewidth=3, label='DP Strategy')
        axes[0,0].plot(greedy_results[0], 'r--', linewidth=3, label='Greedy Strategy')
        
        # Mark DP pit stops with compound change annotations
        for i, pit_lap in enumerate(dp_results[1]):
            pit_time = dp_results[0][pit_lap-1]
            axes[0,0].scatter(pit_lap-1, pit_time, color='blue', s=200,
                              label='DP Pit Stops' if i == 0 else "",
                              zorder=10, marker='o', edgecolors='white', linewidth=3)
            # Add compound change text
            if i < len(dp_strategies):
                axes[0,0].annotate(dp_strategies[i].split(': ')[1], 
                                  xy=(pit_lap-1, pit_time), 
                                  xytext=(8, 15), textcoords='offset points',
                                  fontsize=12, color='blue', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Mark greedy pit stops with compound change annotations
        for i, pit_lap in enumerate(greedy_results[1]):
            pit_time = greedy_results[0][pit_lap-1]
            axes[0,0].scatter(pit_lap-1, pit_time, color='red', s=200,
                              label='Greedy Pit Stops' if i == 0 else "",
                              zorder=10, marker='s', edgecolors='white', linewidth=3)
            # Add compound change text
            if i < len(greedy_strategies):
                axes[0,0].annotate(greedy_strategies[i].split(': ')[1], 
                                  xy=(pit_lap-1, pit_time), 
                                  xytext=(8, -20), textcoords='offset points',
                                  fontsize=12, color='red', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # Auto-scale y-axis to show all lap time variations clearly
        all_times = dp_results[0] + greedy_results[0]
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time
        axes[0,0].set_ylim(min_time - time_range*0.15, max_time + time_range*0.15)
        axes[0,0].set_title('Lap Time Progression (Compound Changes Shown)', fontsize=14)
        axes[0,0].set_xlabel('Lap Number', fontsize=12)
        axes[0,0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # SUBPLOT 2: Cumulative race time (running total)
        cum_dp = np.cumsum(dp_results[0])      # Cumulative sum of DP lap times
        cum_greedy = np.cumsum(greedy_results[0])  # Cumulative sum of greedy lap times
        axes[0,1].plot(cum_dp/60, 'b-', linewidth=3, label='DP Strategy')
        axes[0,1].plot(cum_greedy/60, 'r--', linewidth=3, label='Greedy Strategy')
        axes[0,1].set_title('Cumulative Race Time', fontsize=14)
        axes[0,1].set_xlabel('Lap Number', fontsize=12)
        axes[0,1].set_ylabel('Total Time (minutes)', fontsize=12)
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # SUBPLOT 3: Tire wear progression over race
        axes[1,0].plot(dp_results[2], 'b-', linewidth=3, label='DP Wear')
        axes[1,0].plot(greedy_results[2], 'r--', linewidth=3, label='Greedy Wear')
        # Show wear threshold line
        axes[1,0].axhline(self.wear_threshold, color='orange', linestyle=':', 
                         alpha=0.8, linewidth=3, label=f'Pit Threshold ({self.wear_threshold})')
        axes[1,0].set_title('Tire Wear Progression', fontsize=14)
        axes[1,0].set_xlabel('Lap Number', fontsize=12)
        axes[1,0].set_ylabel('Wear (0-1)', fontsize=12)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # SUBPLOT 4: Fuel consumption over race
        axes[1,1].plot(dp_results[3], 'b-', linewidth=3, label='DP Fuel')
        axes[1,1].plot(greedy_results[3], 'r--', linewidth=3, label='Greedy Fuel')
        axes[1,1].set_title('Fuel Consumption', fontsize=14)
        axes[1,1].set_xlabel('Lap Number', fontsize=12)
        axes[1,1].set_ylabel('Fuel (kg)', fontsize=12)
        axes[1,1].set_ylim(0, self.fuel_capacity + 5)
        axes[1,1].legend(fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        # Adjust spacing between subplots for better readability
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.tight_layout()
        plt.show()

    def plot_dp_compound_analysis(self, all_results):
        """
        Create detailed analysis plot showing how DP strategy performs with different starting compounds.
        
        Args:
            all_results: Dictionary containing simulation results for all starting compounds
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("DP Strategy: Impact of Different Starting Tire Compounds", fontsize=18)
        
        # Color coding for tire compounds
        colors = {'soft': 'red', 'medium': 'orange', 'hard': 'blue'}
        
        # SUBPLOT 1: Lap time comparison for all starting compounds
        for compound in self.tire_compounds:
            lap_times = all_results[compound]['dp'][0]
            pit_laps = all_results[compound]['dp'][1]
            # Plot lap times with compound-specific color
            axes[0,0].plot(lap_times, color=colors[compound], linewidth=3, 
                          label=f'Start: {compound.upper()}')
            # Mark pit stops
            for pit_lap in pit_laps:
                axes[0,0].scatter(pit_lap-1, lap_times[pit_lap-1], 
                                 color=colors[compound], s=150, marker='o', zorder=5,
                                 edgecolors='white', linewidth=2)
        
        axes[0,0].set_title('Lap Time Progression by Starting Compound', fontsize=14)
        axes[0,0].set_xlabel('Lap Number', fontsize=12)
        axes[0,0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # SUBPLOT 2: Cumulative race time comparison
        for compound in self.tire_compounds:
            lap_times = all_results[compound]['dp'][0]
            cumulative = np.cumsum(lap_times) / 60  # Convert to minutes
            axes[0,1].plot(cumulative, color=colors[compound], linewidth=3, 
                          label=f'Start: {compound.upper()}')
        
        axes[0,1].set_title('Cumulative Race Time by Starting Compound', fontsize=14)
        axes[0,1].set_xlabel('Lap Number', fontsize=12)
        axes[0,1].set_ylabel('Total Time (minutes)', fontsize=12)
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # SUBPLOT 3: Tire wear progression comparison
        for compound in self.tire_compounds:
            wear_history = all_results[compound]['dp'][2]
            axes[1,0].plot(wear_history, color=colors[compound], linewidth=3, 
                          label=f'Start: {compound.upper()}')
        
        # Show wear threshold
        axes[1,0].axhline(self.wear_threshold, color='gray', linestyle='--', 
                         alpha=0.8, linewidth=2, label='Wear Threshold')
        axes[1,0].set_title('Tire Wear Progression by Starting Compound', fontsize=14)
        axes[1,0].set_xlabel('Lap Number', fontsize=12)
        axes[1,0].set_ylabel('Wear (0-1)', fontsize=12)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # SUBPLOT 4: Performance summary bar chart
        compounds = list(self.tire_compounds.keys())
        race_times = [all_results[c]['dp_time']/60 for c in compounds]  # Convert to minutes
        pit_counts = [len(all_results[c]['dp'][1]) for c in compounds]
        
        x = np.arange(len(compounds))
        width = 0.5  # Bar width
        
        # Create bars with compound-specific colors
        bars1 = axes[1,1].bar(x, race_times, width, 
                             color=[colors[c] for c in compounds], 
                             alpha=0.7, label='Race Time (min)')
        
        # Add performance metrics as text on bars
        for i, (time, pits) in enumerate(zip(race_times, pit_counts)):
            axes[1,1].text(i, time + 1, f'{pits} stops', 
                          ha='center', va='bottom', fontsize=14, fontweight='bold')
            axes[1,1].text(i, time/2, f'{time:.2f}min', 
                          ha='center', va='center', fontsize=14, fontweight='bold')
        
        axes[1,1].set_title('DP Strategy Performance Summary', fontsize=14)
        axes[1,1].set_xlabel('Starting Tire Compound', fontsize=12)
        axes[1,1].set_ylabel('Total Race Time (minutes)', fontsize=12)
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([c.upper() for c in compounds], fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.tight_layout()
        plt.show()

    def plot_greedy_compound_analysis(self, all_results):
        """
        Create detailed analysis plot showing how Greedy strategy performs with different starting compounds.
        Similar to DP analysis but uses dashed lines and square markers to distinguish strategies.
        
        Args:
            all_results: Dictionary containing simulation results for all starting compounds
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("Greedy Strategy: Impact of Different Starting Tire Compounds", fontsize=18)
        
        colors = {'soft': 'red', 'medium': 'orange', 'hard': 'blue'}
        
        # SUBPLOT 1: Lap time comparison (dashed lines for greedy)
        for compound in self.tire_compounds:
            lap_times = all_results[compound]['greedy'][0]
            pit_laps = all_results[compound]['greedy'][1]
            axes[0,0].plot(lap_times, color=colors[compound], linewidth=3, 
                          linestyle='--', label=f'Start: {compound.upper()}')
            # Mark pit stops with square markers (different from DP circles)
            for pit_lap in pit_laps:
                axes[0,0].scatter(pit_lap-1, lap_times[pit_lap-1], 
                                 color=colors[compound], s=150, marker='s', zorder=5,
                                 edgecolors='white', linewidth=2)
        
        axes[0,0].set_title('Lap Time Progression by Starting Compound', fontsize=14)
        axes[0,0].set_xlabel('Lap Number', fontsize=12)
        axes[0,0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # SUBPLOT 2: Cumulative race time (dashed lines)
        for compound in self.tire_compounds:
            lap_times = all_results[compound]['greedy'][0]
            cumulative = np.cumsum(lap_times) / 60
            axes[0,1].plot(cumulative, color=colors[compound], linewidth=3, 
                          linestyle='--', label=f'Start: {compound.upper()}')
        
        axes[0,1].set_title('Cumulative Race Time by Starting Compound', fontsize=14)
        axes[0,1].set_xlabel('Lap Number', fontsize=12)
        axes[0,1].set_ylabel('Total Time (minutes)', fontsize=12)
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # SUBPLOT 3: Tire wear progression (dashed lines)
        for compound in self.tire_compounds:
            wear_history = all_results[compound]['greedy'][2]
            axes[1,0].plot(wear_history, color=colors[compound], linewidth=3, 
                          linestyle='--', label=f'Start: {compound.upper()}')
        
        axes[1,0].axhline(self.wear_threshold, color='gray', linestyle='-', 
                         alpha=0.8, linewidth=2, label='Wear Threshold')
        axes[1,0].set_title('Tire Wear Progression by Starting Compound', fontsize=14)
        axes[1,0].set_xlabel('Lap Number', fontsize=12)
        axes[1,0].set_ylabel('Wear (0-1)', fontsize=12)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # SUBPLOT 4: Performance summary bar chart
        compounds = list(self.tire_compounds.keys())
        race_times = [all_results[c]['greedy_time']/60 for c in compounds]
        pit_counts = [len(all_results[c]['greedy'][1]) for c in compounds]
        
        x = np.arange(len(compounds))
        width = 0.5
        
        bars1 = axes[1,1].bar(x, race_times, width, 
                             color=[colors[c] for c in compounds], 
                             alpha=0.7, label='Race Time (min)')
        
        # Add text annotations on bars
        for i, (time, pits) in enumerate(zip(race_times, pit_counts)):
            axes[1,1].text(i, time + 1, f'{pits} stops', 
                          ha='center', va='bottom', fontsize=14, fontweight='bold')
            axes[1,1].text(i, time/2, f'{time:.2f}min', 
                          ha='center', va='center', fontsize=14, fontweight='bold')
        
        axes[1,1].set_title('Greedy Strategy Performance Summary', fontsize=14)
        axes[1,1].set_xlabel('Starting Tire Compound', fontsize=12)
        axes[1,1].set_ylabel('Total Race Time (minutes)', fontsize=12)
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([c.upper() for c in compounds], fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.tight_layout()
        plt.show()

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    """
    Main execution block that runs the complete F1 strategy analysis.
    This block orchestrates all simulations and generates comprehensive results.
    """
    
    # Create simulator instance
    simulator = SilverstoneSimulator()
    
    # PHASE 1: Find optimal starting compounds for both strategies
    print("=== FINDING OPTIMAL STARTING COMPOUNDS ===")
    all_results, best_dp_start, best_greedy_start = simulator.find_optimal_starting_compound()
    
    # Display starting compound analysis
    print(f"\nStarting Compound Analysis:")
    for compound in simulator.tire_compounds:
        dp_time = all_results[compound]['dp_time']
        greedy_time = all_results[compound]['greedy_time']
        print(f"{compound.upper()}: DP={dp_time/60:.2f}min, Greedy={greedy_time/60:.2f}min")
    
    print(f"\nOptimal Starting Compounds:")
    print(f"DP Strategy: {best_dp_start.upper()}")
    print(f"Greedy Strategy: {best_greedy_start.upper()}")
    
    # PHASE 2: Run detailed simulations with optimal starting compounds
    dp_results = simulator.simulate_dp_race(best_dp_start)
    greedy_results = simulator.simulate_greedy_race(best_greedy_start)
    
    # PHASE 3: Display detailed results
    print(f"\n=== OPTIMAL STRATEGY RESULTS ===")
    print(f"\nDynamic Programming (Start: {best_dp_start.upper()}):")
    print(f"Total race time: {sum(dp_results[0])/60:.2f} minutes")
    print(f"Pit stops: {len(dp_results[1])} at laps {dp_results[1]}")
    print(f"Compound changes:")
    for strategy in dp_results[4]:
        print(f"  {strategy}")
    print(f"Final fuel: {dp_results[3][-1]:.1f} kg")
    
    print(f"\nGreedy Strategy (Start: {best_greedy_start.upper()}):")
    print(f"Total race time: {sum(greedy_results[0])/60:.2f} minutes")
    print(f"Pit stops: {len(greedy_results[1])} at laps {greedy_results[1]}")
    print(f"Compound changes:")
    for strategy in greedy_results[4]:
        print(f"  {strategy}")
    print(f"Final fuel: {greedy_results[3][-1]:.1f} kg")
    
    # Calculate and display performance comparison
    print(f"\nStrategy Comparison:")
    time_diff = sum(greedy_results[0]) - sum(dp_results[0])
    print(f"DP advantage: {time_diff/60:.2f} minutes")
    
    # PHASE 4: Generate all visualization plots
    
    # Main comparison plot showing DP vs Greedy head-to-head
    simulator.plot_comparison(dp_results, greedy_results, dp_results[4], greedy_results[4])
    
    # Individual strategy analysis plots
    print("\n=== GENERATING COMPOUND ANALYSIS PLOTS ===")
    simulator.plot_dp_compound_analysis(all_results)      # DP strategy with all compounds
    simulator.plot_greedy_compound_analysis(all_results)  # Greedy strategy with all compounds
